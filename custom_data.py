import tensorflow as tf
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class DataPreprocessor:
    def __init__(self, train_csv, job_companies_csv, job_tags_csv, tags_csv, user_tags_csv):
        self.train = pd.read_csv(train_csv) # userID, jobID, applied
        self.job_tags = pd.read_csv(job_tags_csv) # tagID of jobID
        self.user_tags = pd.read_csv(user_tags_csv) # tag of users
        self.tags = pd.read_csv(tags_csv) # actual tag names
        self.job_companies = pd.read_csv(job_companies_csv) # company of jobs

        self.tag_dict = {}
        for i, r in enumerate(self.tags["tagID"]):
            self.tag_dict[r] = i
        # print(len(self.tag_dict.keys()))
        self.refinery()
        self.vectorize()
        # self.vectorize_user_tag()
        print("Processed Data built")

    def refinery(self):
        print("Drop duplicated data")
        print(self.user_tags.shape, self.job_tags.shape, self.tags.shape)
        self.user_tags = self.user_tags.drop_duplicates()
        # no need
        # self.job_tags = self.job_tags.drop_duplicates()
        # self.tags = self.tags.drop_duplicates()
        print(self.user_tags.shape, self.job_tags.shape, self.tags.shape)
        
    def tag_list(self, li):
        res = [0 for _ in self.tags["tagID"]]
        for t in li:
            res[self.tag_dict[t]] = 1

        return res

    def vectorize(self):
        self.job_tags = self.job_tags.groupby("jobID")["tagID"].apply(self.tag_list).reset_index()
        self.user_tags = self.user_tags.groupby("userID")["tagID"].apply(self.tag_list).reset_index()
        def decode(obj):
            if type(obj)==float:
                return [0,0]
            elif "-" in obj:
                return list(map(int, obj.split("-")))
            else :
                return [obj.split(" ")[0], 0]
        self.job_companies["companySize"] = self.job_companies["companySize"].apply(decode)



    def prepro(self, train):
        print("//////////////////////////")
        merged_df = train.merge(self.user_tags, on="userID").merge(self.job_tags, on="jobID", suffixes=('_user', '_job')).merge(self.job_companies[["jobID", "companySize"]], on="jobID")
        print(merged_df.shape)
        print(merged_df.companySize.shape)
        company_size = np.array([arr for arr in merged_df.companySize], int)
        print(company_size[:, 0])
        merged_df = np.array((merged_df.apply(lambda row:[a and b for a, b in zip(row["tagID_user"], row["tagID_job"])], axis=1)).tolist())
        # matched = np.array([sum(row) for row in merged_df]).reshape(-1, 1)
        # merged_df = np.hstack((merged_df, matched))
        merged_df = np.hstack((merged_df, company_size))
        print(merged_df)
        print("Vectorize completed")
        return merged_df    

    def train_start(self):
        X = self.prepro(self.train)
        y = self.train["applied"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        log_reg = LogisticRegression(random_state=23)
        log_reg.fit(X_train, y_train)
        y_pred = log_reg.predict(X_test)
        print ('accur : {:.3f}'.format(accuracy_score(y_test, y_pred)) )


    def check_train(self):
        # train = self.train.groupby("userID").apply(lambda row:list(row[["jobID", "applied"]]))
        # print(train)
        pass
    def calculate_corr(self):
        mdf = self.merged_df
        print(mdf["applied"].corr(mdf["tag_cnt"]))
        print(mdf["applied"].corr(mdf["tag_per_job"]))
        print(mdf["applied"].corr(mdf["tag_per_user"]))
    def prepro_exp(self):
        self.job_tags = self.job_tags.groupby("jobID")["tagID"].apply(lambda li : [self.tag_dict[l] for l in li]).reset_index()
        self.user_tags = self.user_tags.groupby("userID")["tagID"].apply(lambda li : [self.tag_dict[l] for l in li]).reset_index()
        job_tags = self.job_tags
        user_tags = self.user_tags
        train = self.train
        tags = self.tags

        print(job_tags.shape)
        print(user_tags.shape)
        print("//////////////////////////")
        merged_df = train.merge(user_tags, on="userID").merge(job_tags, on="jobID", suffixes=('_user', '_job'))

        print(merged_df.shape)
        # print(merged_df["tagID_job"].head)
        # print(merged_df["tagID_user"].head)
        print(merged_df.columns)
        # return

        merged_df["tag_cnt"] = merged_df.apply(lambda row:(len(set(row["tagID_user"]) & set(row['tagID_job']))), axis=1)
        merged_df["tag_per_job"] = merged_df.apply(lambda row:row["tag_cnt"]/len(row["tagID_job"]), axis=1)
        merged_df["tag_per_user"] = merged_df.apply(lambda row:row["tag_cnt"]/len(row["tagID_user"]), axis=1)

        # print(merged_df[["tag_cnt", "applied"]][merged_df["applied"]==1].head)
        # print(merged_df["tagID_user"].apply(len))
        print(merged_df["applied"].corr(merged_df["tag_cnt"]))
        merged_df.to_csv("./prac.csv")
        print(tags.shape)
        self.merged_df = merged_df
        return
        #experimental
        merged_df["tag_cnt"] = merged_df.apply(lambda row:(sum([1 for i in range(len(self.tags)) if row["tagID_job"][i] * row["tagID_user"][i] != 0])), axis=1)
        merged_df["tag_per_job"] = merged_df.apply(lambda row:row["tag_cnt"]/sum(row["tagID_job"]), axis=1)
        merged_df["tag_per_user"] = merged_df.apply(lambda row:row["tag_cnt"]/sum(row["tagID_user"]), axis=1)

        # print(merged_df[["tag_cnt", "applied"]][merged_df["applied"]==1].head)
        # print(merged_df["tagID_user"].apply(len))
        print(merged_df["applied"].corr(merged_df["tag_cnt"]))
        merged_df.to_csv("./prac.csv")
        print(tags.shape)
        self.merged_df = merged_df


def custom_data(file_path, column_names):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=2,
        label_name="applied"
    )