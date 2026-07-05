"""Inmarsat-C frame parser: turn a decoded 640-byte frame into packets and
readable EGC/marine messages.

A frame is a back-to-back stream of packets. Each packet begins with a
descriptor byte that also encodes its length:

    short  descriptor (0x00-0x7F): length = (descriptor & 0x0F) + 1
    medium descriptor (0x80-0xBF): length = frame[pos+1] + 2

The last two bytes of a packet are a checksum (a Fletcher-style pair, modulo
256). Text payloads are IA5 (7-bit ASCII), ITA2 (Baudot with letter/figure
shifts), or binary. Messages too large for one packet are split with a
multi-frame start (0xBD) and continue (0xBE) and reassembled here.

This is an independent implementation of the packet structure described in the
GPL-3.0 Scytale-C lineage. It is validated structurally: packets built to the
documented format (with correct checksums) parse back with their fields and text
intact (see `_selftest`). Field semantics for live traffic are confirmed only
against a real capture.
"""
from __future__ import annotations

FRAME_LEN = 640

# --- reference tables -------------------------------------------------------
DESCRIPTOR_NAMES = {
    0x27: "Logical Channel Clear", 0x2A: "Inbound Message Ack",
    0x08: "Acknowledgement Request", 0x6C: "Signalling Channel",
    0x7D: "Bulletin Board", 0x81: "Announcement",
    0x83: "Logical Channel Assignment", 0x91: "Distress Alert Ack",
    0x92: "Login Ack", 0x9A: "Enhanced Data Report Ack",
    0xA0: "Distress Test Request", 0xA3: "Individual Poll",
    0xA8: "Confirmation", 0xAA: "Message", 0xAB: "LES List",
    0xAC: "Request Status", 0xAD: "Test Result",
    0xB1: "EGC double header, part 1", 0xB2: "EGC double header, part 2",
    0xBD: "Multiframe Packet Start", 0xBE: "Multiframe Packet Continue",
}

SAT_NAMES = {
    0: "Atlantic Ocean Region West (AOR-W)",
    1: "Atlantic Ocean Region East (AOR-E)",
    2: "Pacific Ocean Region (POR)",
    3: "Indian Ocean Region (IOR)",
    9: "All Ocean Regions Covered by the LES",
}

PRIORITY_NAMES = {-1: "Message", 0: "Routine", 1: "Safety", 2: "Urgency", 3: "Distress"}

SERVICE_NAMES = {
    0x00: "System, All ships (general call)",
    0x02: "FleetNET, Group Call",
    0x04: "SafetyNET, Nav/Met/Piracy Warning to a Rectangular Area",
    0x11: "System, Inmarsat System Message",
    0x13: "SafetyNET, Nav/Met/Piracy Coastal Warning",
    0x14: "SafetyNET, Shore-to-Ship Distress Alert to Circular Area",
    0x23: "System, EGC System Message",
    0x24: "SafetyNET, Nav/Met/Piracy Warning to a Circular Area",
    0x31: "SafetyNET, NAVAREA/METAREA Warning or Forecast",
    0x33: "System, Download Group Identity",
    0x34: "SafetyNET, SAR Coordination to a Rectangular Area",
    0x44: "SafetyNET, SAR Coordination to a Circular Area",
    0x72: "FleetNET, Chart Correction Service",
    0x73: "SafetyNET, Chart Correction Service for Fixed Areas",
}

_ADDRESS_LEN = {0x00: 3, 0x11: 4, 0x31: 4, 0x02: 5, 0x72: 5,
                0x13: 6, 0x23: 6, 0x33: 6, 0x73: 6,
                0x04: 7, 0x14: 7, 0x24: 7, 0x34: 7, 0x44: 7}

PRESENTATION_IA5, PRESENTATION_ITA2, PRESENTATION_BINARY = 0, 6, 7

# --- ITA2 (Baudot) ----------------------------------------------------------
_ITA2_LTRS = {
    0x00: "\0", 0x01: "E", 0x02: "\n", 0x03: "A", 0x04: " ", 0x05: "S",
    0x06: "I", 0x07: "U", 0x08: "\r", 0x09: "D", 0x0A: "R", 0x0B: "J",
    0x0C: "N", 0x0D: "F", 0x0E: "C", 0x0F: "K", 0x10: "T", 0x11: "Z",
    0x12: "L", 0x13: "W", 0x14: "H", 0x15: "Y", 0x16: "P", 0x17: "Q",
    0x18: "O", 0x19: "B", 0x1A: "G", 0x1C: "M", 0x1D: "X", 0x1E: "V",
}
_ITA2_FIGS = {
    0x00: "\0", 0x01: "3", 0x02: "\n", 0x03: "-", 0x04: " ", 0x05: "'",
    0x06: "8", 0x07: "7", 0x08: "\r", 0x0A: "4", 0x0C: ",", 0x0D: "!",
    0x0E: ":", 0x0F: "(", 0x10: "5", 0x11: "+", 0x12: ")", 0x13: "2",
    0x14: "$", 0x15: "6", 0x16: "0", 0x17: "1", 0x18: "9", 0x19: "?",
    0x1A: "&", 0x1C: ".", 0x1D: "/", 0x1E: ";",
}
_ITA2_LTRS_SHIFT, _ITA2_FIGS_SHIFT = 0x1F, 0x1B


def _decode_ita2(codes) -> str:
    out = []
    table = _ITA2_LTRS
    for c in codes:
        c &= 0x1F
        if c == _ITA2_LTRS_SHIFT:
            table = _ITA2_LTRS
        elif c == _ITA2_FIGS_SHIFT:
            table = _ITA2_FIGS
        else:
            out.append(table.get(c, ""))
    return "".join(out)


def _decode_ia5(data) -> str:
    return "".join(chr(b & 0x7F) for b in data)


def _decode_text(data, presentation) -> str:
    if presentation == PRESENTATION_ITA2:
        return _decode_ita2(data)
    return _decode_ia5(data)


# --- checksum ---------------------------------------------------------------
def _crc(frame, pos, length) -> int:
    """Fletcher-style pair over the packet, checksum field treated as zero."""
    c0 = c1 = 0
    for i in range(length):
        b = frame[pos + i] if i < length - 2 else 0
        c0 = (c0 + b) & 0xFFFF
        c1 = (c1 + c0) & 0xFFFF
    cb1 = (c0 - c1) & 0xFF
    cb2 = (c1 - 2 * c0) & 0xFF
    return (cb1 << 8) | cb2


def _packet_length(frame, pos) -> int:
    d = frame[pos]
    if d >> 7 == 0:
        return (d & 0x0F) + 1
    if d >> 6 == 0x02:
        return (frame[pos + 1] + 2) if pos + 1 < len(frame) else (len(frame) - pos)
    return len(frame) - pos          # long descriptor: not modelled, consume rest


def sat_name(sat: int) -> str:
    return SAT_NAMES.get(sat, "Unknown")


# --- per-descriptor field extraction ---------------------------------------
def _extract(frame, pos, length, descriptor, crc_ok) -> dict:
    p = {}
    if not crc_ok:
        return p
    d = descriptor
    if d in (0xB1, 0xB2):                      # EGC message header + text
        mt = frame[pos + 2]
        flags = frame[pos + 3]
        priority = (flags & 0x60) >> 5
        addr_len = _ADDRESS_LEN.get(mt, 3)
        presentation = frame[pos + 7]
        body = pos + 8 + addr_len
        payload = frame[body:pos + length - 2]
        p.update(messageType=mt, service=SERVICE_NAMES.get(mt, "Unknown"),
                 priority=priority, priorityText=PRIORITY_NAMES.get(priority, "Unknown"),
                 isDistress=priority == 3,
                 messageId=(frame[pos + 4] << 8) | frame[pos + 5],
                 packetNo=frame[pos + 6], presentation=presentation,
                 continuation=bool(flags & 0x80),
                 text=_decode_text(payload, presentation))
    elif d == 0xAA:                            # message payload
        sat = (frame[pos + 2] >> 6) & 0x03
        payload = frame[pos + 5:pos + length - 2]
        binary = _is_binary(payload)
        p.update(sat=sat, satName=sat_name(sat), lesId=frame[pos + 2] & 0x3F,
                 logicalChannelNo=frame[pos + 3], packetNo=frame[pos + 4],
                 presentation=PRESENTATION_BINARY if binary else PRESENTATION_IA5,
                 text="" if binary else _decode_ia5(payload))
    elif d == 0x7D and length >= 12:           # bulletin board (NCS)
        frame_no = (frame[pos + 2] << 8) | frame[pos + 3]
        secs = frame_no * 8.64
        sat = (frame[pos + 7] >> 6) & 0x03
        p.update(networkVersion=frame[pos + 1], frameNumber=frame_no,
                 sat=sat, satName=sat_name(sat), lesId=frame[pos + 7] & 0x3F,
                 timeOfDay="%02d:%02d:%06.3f" % (int(secs // 3600) % 24,
                                                 int(secs // 60) % 60, secs % 60))
    elif d in (0x27, 0x08, 0x81, 0x83):        # signalling with sat/LES
        off = {0x27: 4, 0x08: 1, 0x81: 5, 0x83: 5}[d]
        sat = (frame[pos + off] >> 6) & 0x03
        p.update(sat=sat, satName=sat_name(sat), lesId=frame[pos + off] & 0x3F)
    return p


def _is_binary(data) -> bool:
    control = set(range(0x01, 0x20)) - {0x02, 0x03, 0x04, 0x09, 0x0A, 0x0D}
    for b in data[:13]:
        c = b & 0x7F
        if c in control or c == ord("$"):
            return True
    return False


# --- frame walker (stateful for multi-frame messages) -----------------------
class Parser:
    """Parses successive frames, reassembling multi-frame packets."""

    def __init__(self):
        self._mfp = None          # (target_len, bytearray, filled)

    def parse_frame(self, frame) -> list[dict]:
        frame = bytes(frame)
        if len(frame) != FRAME_LEN:
            raise ValueError(f"frame must be {FRAME_LEN} bytes")
        packets = []
        pos = 0
        while pos < FRAME_LEN:
            d = frame[pos]
            if d == 0x00:                      # filler: no more packets
                break
            length = _packet_length(frame, pos)
            if length < 3 or pos + length > FRAME_LEN:
                break
            crc = (frame[pos + length - 2] << 8) | frame[pos + length - 1]
            crc_ok = crc == 0 or crc == _crc(frame, pos, length)
            pkt = {"descriptor": d, "name": DESCRIPTOR_NAMES.get(d, "Unknown"),
                   "length": length, "crc_ok": crc_ok, "pos": pos}
            pkt.update(self._handle(frame, pos, length, d, crc_ok))
            packets.append(pkt)
            pos += length
        return packets

    def _handle(self, frame, pos, length, d, crc_ok) -> dict:
        if d == 0xBD and crc_ok:               # multi-frame start
            inner = frame[pos + 2]
            if inner >> 7 == 0:
                target = (inner & 0x0F) + 1
            elif inner >> 6 == 0x02:
                target = frame[pos + 3] + 2
            else:
                target = 0
            first = frame[pos + 2:pos + length - 2]
            self._mfp = [target, bytearray(first)]
            return {"multiframe": "start", "innerDescriptor": inner}
        if d == 0xBE and crc_ok and self._mfp is not None:
            self._mfp[1].extend(frame[pos + 2:pos + length - 2])
            if len(self._mfp[1]) >= self._mfp[0] - 2 > 0:
                assembled = bytes(self._mfp[1]).ljust(FRAME_LEN, b"\x00")[:FRAME_LEN]
                self._mfp = None
                inner = Parser().parse_frame(assembled)
                return {"multiframe": "complete", "packets": inner}
            return {"multiframe": "continue"}
        return _extract(frame, pos, length, d, crc_ok)


def parse_frame(frame) -> list[dict]:
    """Parse a single 640-byte frame (no multi-frame state kept)."""
    return Parser().parse_frame(frame)


def messages(packets) -> list[dict]:
    """Filter a packet list down to the ones carrying readable text."""
    out = []
    for pkt in packets:
        if pkt.get("text"):
            out.append(pkt)
        for inner in pkt.get("packets", []):
            if inner.get("text"):
                out.append(inner)
    return out


def render(packets) -> str:
    """Human-readable one-line-per-packet summary of a parsed frame."""
    lines = []
    for pkt in packets:
        tag = "ok " if pkt["crc_ok"] else "BAD"
        line = f"[{tag}] {pkt['descriptor']:02X} {pkt['name']}"
        if "satName" in pkt:
            line += f"  sat={pkt['satName']}"
        if "service" in pkt:
            line += f"  {pkt['service']} ({pkt['priorityText']})"
            if pkt.get("isDistress"):
                line += " *** DISTRESS ***"
        lines.append(line)
        if pkt.get("text"):
            body = pkt["text"].replace("\r", "").strip()
            for tline in body.splitlines():
                lines.append("      | " + tline)
    return "\n".join(lines)


# --- self-test --------------------------------------------------------------
def _build_medium_packet(descriptor, body) -> bytes:
    """Assemble a medium-descriptor packet with a correct checksum."""
    length = 2 + len(body) + 2                 # desc + len byte + body + CRC
    pkt = bytearray([descriptor, length - 2]) + bytearray(body) + bytearray(2)
    crc = _crc(pkt, 0, length)
    pkt[-2], pkt[-1] = crc >> 8, crc & 0xFF
    return bytes(pkt)


def build_egc(message_type, priority, text, presentation=PRESENTATION_IA5,
              message_id=0x1234, packet_no=1) -> bytes:
    """Assemble one EGC message packet (0xB1) with a valid checksum. Mainly for
    tests and the offline demonstration."""
    addr_len = _ADDRESS_LEN.get(message_type, 3)
    header = bytes([message_type, (priority & 0x03) << 5,
                    (message_id >> 8) & 0xFF, message_id & 0xFF,
                    packet_no & 0xFF, presentation])
    body = header + bytes(addr_len) + text.encode("latin-1", "ignore")
    return _build_medium_packet(0xB1, body)


def build_frame(packets) -> bytes:
    """Pack packets into a 640-byte frame padded with filler."""
    blob = b"".join(packets)
    if len(blob) > FRAME_LEN:
        raise ValueError("packets exceed frame length")
    return blob.ljust(FRAME_LEN, b"\x00")


def _selftest() -> dict:
    text = "TEST NAVAREA III 001/26 GALE WARNING"
    # EGC message (0xB1): messageType 0x31 (NAVAREA, addr len 4), priority urgency.
    mt, addr_len = 0x31, 4
    body = bytes([mt, 0x40, 0x12, 0x34, 0x01, PRESENTATION_IA5]) + bytes(addr_len) + text.encode()
    egc = _build_medium_packet(0xB1, body)
    # Message payload (0xAA) with IA5 text.
    aa_text = "SHIP MV EXAMPLE POS N37.5 E013.2"
    aa_body = bytes([0xC0 & 0xC0, 0x05, 0x01]) + aa_text.encode()  # sat/les, chan, pkt
    aa = _build_medium_packet(0xAA, aa_body)

    frame = (egc + aa).ljust(FRAME_LEN, b"\x00")
    packets = parse_frame(frame)
    msgs = messages(packets)
    found = {m.get("text", "") for m in msgs}
    return {
        "packets": len(packets),
        "all_crc_ok": all(p["crc_ok"] for p in packets if p["descriptor"] != 0),
        "egc_text_ok": text in found,
        "aa_text_ok": aa_text in found,
        "distress_flagged": any(p.get("isDistress") for p in packets),
    }


if __name__ == "__main__":
    r = _selftest()
    print(r)
