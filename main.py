from custom_data import *



def start():
    data_root = "./data/"
    train_root = data_root + "train_job/"
    C = DataPreprocessor(train_root+"train.csv", train_root+"job_companies.csv", train_root+"job_tags.csv", train_root+"tags.csv", train_root+"user_tags.csv")
    # C.prepro()
    # C.calculate_corr()
    # X_train, X_test, y_train, y_test = C.data_split()
    C.train_start()
    pass

if __name__ == "__main__":
    start()