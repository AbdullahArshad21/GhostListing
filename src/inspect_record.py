import tensorflow as tf

SHARD_PATH = r"C:\Users\TOP WAY\Downloads\archive (2)\stanford_online_products\1.0.0\stanford_online_products-test.tfrecord-00000-of-00016"

raw_dataset = tf.data.TFRecordDataset(SHARD_PATH)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        values = getattr(feature, kind).value
        print(f"key: {key!r}")
        print(f"  type: {kind}")
        print(f"  length: {len(values)}")
        print(f"  first few values: {list(values[:5])}")
        print()