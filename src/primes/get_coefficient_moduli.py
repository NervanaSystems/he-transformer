import numpy as np
import os
import zipfile
import argparse


# Used to generate valid coefficient moduli for SEAL configuration
def main(FLAGS):

    all_primes = []

    print('loading 50 million primes:')
    for i in range(1, 51):
        print(i, end=' ', flush=True)
        zip_filename = './primes' + str(i) + '.zip'
        txt_filename = './primes' + str(i) + '.txt'
        url = 'https://primes.utm.edu/lists/small/millions/primes' + str(
            i) + '.zip'

        if not os.path.isfile(zip_filename):
            os.system('wget ' + url + ' --no-check-certificate')

        primes = np.load(zip_filename)
        primes = primes[txt_filename[2:]]
        primes = primes.split()
        # Skip header
        primes = primes[6:]
        primes = [int(prime) for prime in primes]
        all_primes.extend(primes)

    print(len(all_primes))
    # print(len(set(all_primes)))
    all_primes = sorted(all_primes, reverse=True)

    modulus_primes = []
    for prime in all_primes:
        if (prime % (2 * FLAGS.N) == 1):
            bits = np.log2(prime)
            if bits > FLAGS.low and bits < FLAGS.high:
                modulus_primes.append(prime)
    print('modulus_primes', modulus_primes)

    modulus_primes = sorted(modulus_primes)

    for prime in modulus_primes:
        print(prime, np.log2(prime), 'bits, hex ', hex(prime))

    #print(' primes', modulus_primes)
    #print('bits', [np.log2(x) for x in modulus_primes])
    #print('hex', [hex(x) for x in modulus_primes])
    #ratios = [x / float(y) for x, y in zip(modulus_primes, modulus_primes[1:])]
    # print('ratios', ratios)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--N', type=int, default=2048, help='polynomial modulus degree')
    parser.add_argument(
        '--low',
        type=float,
        default=20,
        help='smallest number of bits in modulus')
    parser.add_argument(
        '--high',
        type=float,
        default=21,
        help='largest number of bits in modulus')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
