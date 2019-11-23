import json


count_data = {}
total_count = 0
train_count = 0
dev_count = 0
test_count = 0


def count(filename, key):
    global count_data
    global train_count
    global dev_count
    global test_count
    global total_count
    with open(filename) as file:
        for json_data in file:
            obj = json.loads(json_data)
            source = obj["source"]
            if source in count_data:
                count_data[source] += 1
            else:
                count_data[source] = 1
            total_count += 1
            if key == "train":
                train_count += 1
            elif key == "dev":
                dev_count += 1
            elif key == "test":
                test_count += 1
            else:
                print("This shouldn't be happening.")


def display():
    global count_data
    global train_count
    global dev_count
    global test_count
    global total_count

    print("\n\n")
    print("Total Data: ", total_count)
    print("Total Training Data: ", train_count)
    print("Total Validation Data: ", dev_count)
    print("Total Test Data: ", test_count)
    print("\n\n")

    for source in count_data:
        print("{}: {}".format(source, count_data[source]))
    print("\n\n")


count("disaster_tweet_id_train.json", "train")
count("disaster_tweet_id_dev.json", "dev")
count("disaster_tweet_id_test.json", "test")

display()
