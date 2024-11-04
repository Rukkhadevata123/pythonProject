def gcd(a: int, b: int) -> int:
    return a if b == 0 else gcd(b, a % b)


def is_even(a: int) -> bool:
    return a & 1 == 0


def generate_pythagorean_triples(_x: int) -> list[tuple[int, ...]]:
    _triples = []
    for i in range(1, _x):
        for u in range(i + 1, _x + 1):
            if gcd(i, u) == 1 and (not is_even(i + u)):
                q = 2 * i * u
                w = abs(u * u - i * i)
                c = u * u + i * i
                sb = sorted([q, w, c])
                _triples.append(tuple(sb))

    _triples.sort(key=lambda _triple: (_triple[0], _triple[1]))
    return _triples


if __name__ == "__main__":
    x = 10
    triples = generate_pythagorean_triples(x)
    for triple in triples:
        print(triple)
