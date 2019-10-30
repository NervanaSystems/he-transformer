# Train the network
First, train the network using
```bash
python example.py [--epochs=12 --batch_size=128]
```
This trains the network briefly and stores the network weights.


# Test the network
To test the network using nGraph's CPU backend
```bash
python example.py --test=true [--batch_size=128]
```

# HE plaintext data
To run using he-transformer,
```bash
python example.py \
--test=true \
--backend=HE_SEAL \
--encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
```

# HE Encrypted data
