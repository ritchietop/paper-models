import tensorflow as tf


class TitanicData:
    @staticmethod
    def input_fn():
        files_pattern = "/home/mi/Projects/Self/paper-models/examples/data/titanic/train.csv"
        column_names = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                        "Cabin", "Embarked"]
        column_defaults = [-1, 0, 0, "", "", -1.0, -1, -1, "", 0.0, "", ""]

        def convert(features, label):
            features["CabinType"] = tf.strings.substr(features["Cabin"], pos=0, len=1)
            features["TicketTypeTmp"] = tf.strings.regex_replace(features["Ticket"], pattern="\.*\s*\d+$", rewrite="")
            features["TicketType"] = tf.strings.regex_replace(features["TicketTypeTmp"], pattern="[/\.\s]", rewrite="")
            del features["TicketTypeTmp"]
            return features, label

        data = tf.data.experimental.make_csv_dataset(
            file_pattern=files_pattern,
            batch_size=200,
            column_names=column_names,
            column_defaults=column_defaults,
            label_name="Survived",
            select_columns=None,
            field_delim=",",
            use_quote_delim=True,
            na_value="",
            header=True,
            num_epochs=1,
            shuffle=False,
            shuffle_buffer_size=1000,
            prefetch_buffer_size=-1,
            num_rows_for_inference=100,
            compression_type=None,
            ignore_errors=False)

        return data.map(convert)

    def get_feature_columns(self):
        p_class = tf.feature_column.categorical_column_with_vocabulary_list(key="Pclass", vocabulary_list=[1, 2, 3])
        sex = tf.feature_column.categorical_column_with_vocabulary_list(key="Sex", vocabulary_list=["male", "female"])
        age = tf.feature_column.bucketized_column(
            source_column=tf.feature_column.numeric_column(key="Age", dtype=tf.float32),
            boundaries=[0, 1, 6, 16])
        sibsp = tf.feature_column.categorical_column_with_vocabulary_list(
            key="SibSp", vocabulary_list=[0, 1, 2, 3, 4, 5, 6, 8])
        parch = tf.feature_column.categorical_column_with_identity(key="Parch", num_buckets=7)
        ticket = tf.feature_column.categorical_column_with_hash_bucket(key="Ticket", hash_bucket_size=10)
        fare = tf.feature_column.numeric_column(key="Fare", dtype=tf.float32)
        cabin_type = tf.feature_column.categorical_column_with_vocabulary_list(
            key="CabinType", vocabulary_list=["A", "B", "C", "D", "E", "F", "G", "T"])
        cabin = tf.feature_column.categorical_column_with_hash_bucket(key="Cabin", hash_bucket_size=10)
        embarked = tf.feature_column.categorical_column_with_vocabulary_list(
            key="Embarked", vocabulary_list=["S", "C", "Q"])

    def get_dataset(batch_size):
        train_path = "./data/titanic/train.csv"
        train_column_names = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket",
                              "Fare",
                              "Cabin", "Embarked"]
        train_column_defaults = ["", 0, 0, "", "", 0.0, -1, -1, "", 0.0, "", ""]
        train_data = tf.data.experimental.make_csv_dataset(file_pattern=train_path, batch_size=batch_size, num_epochs=1,
                                                           column_names=train_column_names,
                                                           column_defaults=train_column_defaults, label_name="Survived",
                                                           num_rows_for_inference=0)

        test_path = "./data/titanic/test.csv"
        test_column_names = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare",
                             "Cabin", "Embarked"]
        test_column_defaults = ["", 0, "", "", 0.0, -1, -1, "", 0.0, "", ""]
        test_data = tf.data.experimental.make_csv_dataset(file_pattern=test_path, batch_size=batch_size, num_epochs=1,
                                                          column_names=test_column_names,
                                                          column_defaults=test_column_defaults,
                                                          num_rows_for_inference=0)

        return train_data, test_data

    def feature_processing(features):
        pclass_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=[1, 2, 3],
                                                                                  num_oov_indices=0)
        pclass_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="Pclass")
        sex_process = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=['male', 'female'],
                                                                              num_oov_indices=0)
        sex_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Sex")
        age_process = tf.keras.layers.experimental.preprocessing.Discretization(bins=[6, 12, 18, 30, 40, 50, 60])
        age_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="Age")
        sibsp_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(9)),
                                                                                 num_oov_indices=0)
        sibsp_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="SibSp")
        parch_process = tf.keras.layers.experimental.preprocessing.IntegerLookup(vocabulary=list(range(7)),
                                                                                 num_oov_indices=0)
        parch_input = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="Parch")
        ticket_process = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
        ticket_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Ticket")
        fare_process = tf.keras.layers.Lambda(function=lambda tensor: tf.math.log1p(tensor))
        fare_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name="Fare")
        cabin_process = tf.keras.layers.experimental.preprocessing.Hashing(num_bins=200)
        cabin_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Cabin")
        embarked_process = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=["S", "C"],
                                                                                   num_oov_indices=0)
        embarked_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name="Embarked")

    from sklearn.preprocessing import OrdinalEncoder


class MovieLensData:
    @staticmethod
    def get_base_dir():
        file_path = os.path.abspath(__file__)
        return file_path[:file_path.rindex("/")]

    @staticmethod
    def input_fn(batch_size, num_epochs, label_key="rating"):
        return tf.data.experimental.make_batched_features_dataset(
            file_pattern="{}/ml-1m/data.tfrecord".format(DataSourceUtils.get_base_dir()),
            batch_size=batch_size,
            features=DataSourceUtils.get_feature_schema(),
            reader=tf.data.TFRecordDataset,
            label_key=label_key,
            num_epochs=num_epochs)

    @staticmethod
    def get_feature_schema():
        return {
            "user_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
            "user_gender": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value="null"),
            "user_age": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
            "user_occupation": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
            "user_zip_code": tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value="null"),
            "user_random_movie_ids": tf.io.VarLenFeature(dtype=tf.int64),
            "movie_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
            "movie_publish_year": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
            "movie_genres": tf.io.VarLenFeature(dtype=tf.string),
            "rating": tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0),
            "rating_n": tf.io.FixedLenFeature(shape=(3,), dtype=tf.float32, default_value=[0, 0, 0]),
            "label": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0)
        }

    @staticmethod
    def get_categorical_columns():
        return [
            tf.feature_column.categorical_column_with_vocabulary_list(key="user_gender", vocabulary_list=["F", "M"]),
            tf.feature_column.categorical_column_with_vocabulary_list(
                key="user_age", vocabulary_list=[1, 18, 25, 35, 45, 50, 56]),
            tf.feature_column.categorical_column_with_identity(key="user_occupation", num_buckets=21),
            tf.feature_column.categorical_column_with_vocabulary_list(
                key="movie_genres", vocabulary_list=["Action", "Adventure", "Animation", "Children's", "Comedy",
                                                     "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                                                     "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
                                                     "Western"])
        ]

    @staticmethod
    def get_dense_columns():
        return [
            tf.feature_column.embedding_column(
                categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
                    key="user_zip_code", hash_bucket_size=1000),
                dimension=10,
                combiner="mean")
        ]

    @staticmethod
    def generate(output_path):
        user_info = DataSourceUtils.load_user()
        movie_info = DataSourceUtils.load_movie()
        ratings = DataSourceUtils.load_rating()
        writer = tf.io.TFRecordWriter(output_path)
        all_movie_ids = list(movie_info.keys())
        for user_id, movie_id, rating in ratings:
            example = tf.train.Example(features=tf.train.Features(feature={
                "user_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
                "user_gender": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[user_info[user_id]["gender"].encode("utf-8")])),
                "user_age": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[user_info[user_id]["age"]])),
                "user_occupation": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[user_info[user_id]["occupation"]])),
                "user_zip_code": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[user_info[user_id]["zip_code"].encode("utf-8")])),
                "user_random_movie_ids": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=random.sample(all_movie_ids, k=random.randint(0, 10)))),
                "movie_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[movie_id])),
                "movie_publish_year": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[movie_info[movie_id]["publish_year"]])),
                "movie_genres": tf.train.Feature(
                    bytes_list=tf.train.BytesList(
                        value=list(map(lambda x: x.encode("utf-8"), movie_info[movie_id]["genres"])))),
                # 回归问题的预测值
                "rating": tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
                "rating_n": tf.train.Feature(float_list=tf.train.FloatList(value=[rating, rating + 1, rating + 2])),
                # 这里简单使用3作为分割值，来划分正例负例
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1 if rating >= 3 else 0]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    @staticmethod
    def load_user():
        user_info = defaultdict(dict)
        with open("{}/ml-1m/users.dat".format(DataSourceUtils.get_base_dir()), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                id, gender, age, occupation, zip_code = line.strip("\r\n").split("::")
                user_info[int(id)] = {
                    "gender": gender,
                    "age": int(age),
                    "occupation": int(occupation),
                    "zip_code": zip_code
                }
        return user_info

    @staticmethod
    def load_movie():
        movie_info = defaultdict(dict)
        with open("{}/ml-1m/movies.dat".format(DataSourceUtils.get_base_dir()), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                id, title, genres = line.strip("\r\n").split("::")
                publish_year = title[title.rindex("(") + 1:-1]
                movie_info[int(id)] = {
                    "publish_year": int(publish_year),
                    "genres": genres.split("|")
                }
        return movie_info

    @staticmethod
    def load_rating():
        ratings = []
        with open("{}/ml-1m/ratings.dat".format(DataSourceUtils.get_base_dir()), "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                user_id, movie_id, rating, _ = line.strip("\r\n").split("::")
                ratings.append((int(user_id), int(movie_id), float(rating)))
        return ratings
