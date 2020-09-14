import pandas as pd
import numpy as np
# import seaborn as sns
#sns.set(style="ticks",color_codes=True)
# import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')

class Show_Accuracy():
    
    def __init__(self, result_file=None, result_df=None):
        self.result_file = result_file
        self.result_df = result_df
    
    def calc_diff_ratio(self,pred, true):
        pred = float(pred)
        true = float(true)
        return (pred - true)/true

    def show(self):
        #result_file = "Results/online-result.csv"
        if self.result_file:
            df_result = pd.read_csv(self.result_file, usecols=['_id', 'Lp', 'Pp', 'Sp'])
        else:
            df_result = self.result_df
            df_result = df_result[['_id', 'Lp', 'Pp', 'Sp']]
            
        df_result.drop_duplicates(inplace=True,keep='last')
        df_result.dropna(inplace=True)
        df_result['diff_ratio'] = df_result.apply(lambda x: abs(self.calc_diff_ratio(x.Pp, x.Sp)), axis = 1)

        # data = df_result['diff_ratio']
        # sns.set(style="ticks",color_codes=True)
        # #sns.set(color_codes=True)
        # ax = sns.kdeplot(data)
        # plt.show()
        # #print(df_result)

        df_percent = pd.DataFrame(columns=['percent', 'count', 'percentage'])
        i = 1
        total_counts = len(df_result)
        while i < 101:
            #print(i)
            df_i = df_result[(df_result.diff_ratio <= i/100) & (df_result.diff_ratio > (i-1)/100)]
            df_percent.loc[-1] = [i, len(df_i), len(df_i)/total_counts]  # adding a row
            df_percent.index = df_percent.index + 1  # shifting index
            df_percent = df_percent.sort_index() 
            i = i + 1
        df_percent.sort_values(by='percent', ascending=True, inplace=True)
        df_percent.reset_index(drop=True, inplace=True)
        #print(df_percent)

        # df_percent_10 = df_percent[(df_percent.percent <= 10) & (df_percent.percent > 0)]
        # sns.barplot(df_percent_10['percent'], 100*df_percent_10['percentage'], palette="BuPu_r")
        # plt.title('Accuracy')
        # plt.ylabel('Per %')
        # sns.despine(bottom=True)
        # plt.show()

        # df_percent_20 = df_percent[(df_percent.percent <= 20) & (df_percent.percent > 10)]
        # sns.barplot(df_percent_20['percent'], 100*df_percent_20['percentage'], palette="BuPu_r")
        # plt.title('Accuracy')
        # plt.ylabel('Per %')
        # sns.despine(bottom=True)
        # plt.show()

        # df_percent_30 = df_percent[(df_percent.percent <= 30) & (df_percent.percent > 20)]
        # sns.barplot(df_percent_30['percent'], 100*df_percent_30['percentage'], palette="BuPu_r")
        # plt.title('Accuracy')
        # plt.ylabel('Per %')
        # sns.despine(bottom=True)
        # plt.show()

        # df_percent_40 = df_percent[(df_percent.percent <= 40) & (df_percent.percent > 30)]
        # sns.barplot(df_percent_40['percent'], 100*df_percent_40['percentage'], palette="BuPu_r")
        # plt.title('Accuracy')
        # plt.ylabel('Per %')
        # sns.despine(bottom=True)
        # plt.show()

        Home_Price_Estimate = len(df_result)
        df_result = df_result.sort_values(by = 'diff_ratio', ascending = False)
        #print(df_result)   
        median_error = df_result['diff_ratio'].quantile(0.5)

        df_result_under_3 = df_result[(df_result.diff_ratio < 0.03)]
        within_3_percent = df_result_under_3.shape[0]/df_result.shape[0]
        median_within_3_percent = df_result_under_3['diff_ratio'].median()   
        df_result_under_5 = df_result[(df_result.diff_ratio < 0.05)]
        within_5_percent = df_result_under_5.shape[0]/df_result.shape[0]
        #mean_within_5_percent = df_result_under_5['diff_ratio'].mean()
        median_within_5_percent = df_result_under_5['diff_ratio'].median()
        df_result_under_10 = df_result[(df_result.diff_ratio < 0.1)]
        within_10_percent = df_result_under_10.shape[0]/df_result.shape[0]
        #mean_within_10_percent = df_result_under_10['diff_ratio'].mean()
        median_within_10_percent = df_result_under_10['diff_ratio'].median()
        df_result_under_20 = df_result[(df_result.diff_ratio < 0.2)]
        within_20_percent = df_result_under_20.shape[0]/df_result.shape[0]
        #mean_within_20_percent = df_result_under_20['diff_ratio'].mean()
        median_within_20_percent = df_result_under_20['diff_ratio'].median()
        df_result_exceed_20 =  df_result[(df_result.diff_ratio > 0.2)]
        mean_exceed_20 = df_result_exceed_20['diff_ratio'].mean()
        median_exceed_20 = df_result_exceed_20['diff_ratio'].median()
        max_exceed_20 = df_result_exceed_20['diff_ratio'].max()

        df_result_outlier = df_result[(df_result.diff_ratio > 0.2)]
        print("Outlier:")
        print(df_result_outlier)

        #df_result_exceed_10 = df_result[(df_result.diff_ratio > 0.1)]
        #print(df_result_exceed_10)
        #df_result_exceed_10.to_csv('result_exceed_10.csv')

        mean_error = df_result['diff_ratio'].mean()

        print("\nPerformance:\n")
        print('Home_Price_Estimate: ' + str(Home_Price_Estimate))
        print("within_3_percent: " + '%.2f%%' % (within_3_percent * 100) + ' with median: ' + '%.2f%%' % (median_within_3_percent * 100)) 
        #print('within_5_percent: ' + '%.2f%%' % (within_5_percent * 100) + ' with mean: ' + '%.2f%%' % (mean_within_5_percent * 100)) 
        print('within_5_percent: ' + '%.2f%%' % (within_5_percent * 100) + ' with median: ' + '%.2f%%' % (median_within_5_percent * 100)) 
        #print('within_10_percent: ' + '%.2f%%' % (within_10_percent * 100) + ' with mean: ' + '%.2f%%' % (mean_within_10_percent * 100)) 
        print('within_10_percent: ' + '%.2f%%' % (within_10_percent * 100) + ' with median: ' + '%.2f%%' % (median_within_10_percent * 100)) 
        #print('within_20_percent: ' + '%.2f%%' % (within_20_percent * 100) + ' with mean: ' + '%.2f%%' % (mean_within_20_percent * 100)) 
        print('within_20_percent: ' + '%.2f%%' % (within_20_percent * 100) + ' with median: ' + '%.2f%%' % (median_within_20_percent * 100)) 
        print('median_error: ' + '%.2f%%' % (median_error * 100))
        print('mean_error: ' + '%.2f%%' % (mean_error * 100)) 
        print('mean_exceed_20: ' + '%.2f%%' % (mean_exceed_20 * 100))
        print('median_exceed_20: ' + '%.2f%%' % (median_exceed_20 * 100))
        print('max_exceed_20: ' + '%.2f%%' % (max_exceed_20 * 100))
        print('.........................................................................................................................')