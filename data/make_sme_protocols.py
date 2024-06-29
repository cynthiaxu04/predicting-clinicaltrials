import pandas as pd

if __name__ == "__main__":
    filename = 'cleaned_data_train.csv'

    df = pd.read_csv(filename)

    n_intervals = 5

    def sample_rows(df, group_col, n=1):
        sampled_df = df.groupby(group_col).apply(lambda x: x.sample(n=n, random_state=42)).reset_index(drop=True)
        return sampled_df


    protocols = pd.DataFrame()
    for i in range(n_intervals):
        sampled_row = sample_rows(df, 'study_eq_labels', n=1)
        protocols = pd.concat([sampled_row], ignore_index=True)

    cols = [
        'protocolSection_identificationModule_nctId',
        'primary_study_duration_days',
        'study_duration_days',
        # 'study_eq_bins',
        # 'primary_eq_bins',
        'study_eq_labels'
        # 'primary_eq_labels'
    ]
    print(df.columns)
    protocols = df[cols]

    protocols['url'] = 'https://clinicaltrials.gov/study/'+protocols['protocolSection_identificationModule_nctId']
    protocols['study_duration_months'] = protocols['study_duration_days']/30
    protocols['study_duration_years'] = protocols['study_duration_months']/12

    protocols.to_csv("sme_protocols.csv", index=False)

    print("File generated.")