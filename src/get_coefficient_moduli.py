import numpy as np
import os
import zipfile
import argparse


# Used to generate valid coefficient moduli for SEAL configuration
def main(FLAGS):

    if not os.path.isfile('./primes1.zip'):
        os.system(
            'wget http://www.utm.edu/~caldwell/primes/millions/primes1.zip')

    if not os.path.isfile('./primes.txt'):
        zip_ref = zipfile.ZipFile('./primes1.zip', 'r')
        zip_ref.extractall('./')
        zip_ref.close()

    primes = np.loadtxt('./primes1.txt', skiprows=2, dtype=int).flatten()

    primes = sorted(primes, reverse=True)
    print(len(primes))

    modulus_primes = []

    for prime in primes:
        if (prime % (2 * FLAGS.N) == 1):
            modulus_primes.append(prime)
    print('modulus_primes', modulus_primes)

    bit_primes = []
    for mod_prime in modulus_primes:
        bits = np.log2(mod_prime)
        if bits > FLAGS.bits - 1 and bits < FLAGS.bits:
            bit_primes.append(mod_prime)

    print(FLAGS.bits, ' primes', bit_primes)

    print('hex', [hex(x) for x in bit_primes])

    ratios = [x / float(y) for x, y in zip(bit_primes, bit_primes[1:])]

    print('ratios', ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--N', type=int, default=2048, help='polynomial modulus degree')
    parser.add_argument(
        '--bits',
        type=int,
        default=20,
        help='number of bits in coefficient modulus')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
