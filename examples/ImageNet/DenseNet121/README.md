

# To test on unencrypted data:
```bash
python test.py \
--data_dir=$DATA_DIR \
--batch_size=10 \
--image_size=224 \
--ngraph=True \
--backend=HE_SEAL
```

# To test on encrypted data
```bash
python test.py \
--data_dir=$DATA_DIR \
--batch_size=1 \
--image_size=224 \
--ngraph=True \
--backend=HE_SEAL \
--encrypt_server_data=yes
```
Note: currently doesn't actually encrypt the data!