import struct

SAMPLE_RATE_HZ = 100
SAMPLES_PER_PACKET = 4
PERIOD = SAMPLES_PER_PACKET / SAMPLE_RATE_HZ

PACKET_FMT = '<' + 'H' + 'HH'*SAMPLES_PER_PACKET
PACKET_SIZE = struct.calcsize(PACKET_FMT)

U16_MAX = 65535
SEQ_MOD = 65536

SCALE_ACC = 1000
SCALE_GYRO = 10



assert PACKET_SIZE == 18, f"Packet size is not {PACKET_SIZE} bytes."

def to_u16(value, scale):
    return round(max(0, min(value * scale, U16_MAX)))



def pack_packet(seq: int, values: list[tuple[float, float]]) -> bytes:
    if len(values) != SAMPLES_PER_PACKET:
        raise ValueError(f"The No. of Samples received is not {SAMPLES_PER_PACKET}.")
    
    data_array = [seq % SEQ_MOD]
    for (a, g) in values:
        a_scaled = to_u16(a, SCALE_ACC)
        g_scaled = to_u16(g, SCALE_GYRO)
        data_array.append(a_scaled)
        data_array.append(g_scaled)
    
    return struct.pack(PACKET_FMT, *data_array)



def parse_packet(packet: bytes) -> tuple[int, list[tuple[float, float]]]:

    if len(packet) != PACKET_SIZE:
        raise ValueError("The size of the Data Packet to be parsed is not 18 bytes.")

    data = struct.unpack(PACKET_FMT, packet)

    data_array = [data[0]]
    values = []
    for i in range(1, len(data) - 1, 2):
        values.append((data[i] / SCALE_ACC, data[i + 1] / SCALE_GYRO))
    data_array.append(values)

    return tuple(data_array)

if __name__ == "__main__":
    data = (65536, [(1.0, 0.0)] * 4)
    packet = pack_packet(*data)
    print(f"data to be packed: {data}")
    print(f"packed data: {packet.hex(' ')}")
    print(f"data after unpacking: {parse_packet(packet)}")

