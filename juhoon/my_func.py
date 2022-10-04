

import numpy as np
import pandas as pd


##### 가로로 길때 전부 살펴보기 ----- (col=N개 컬럼씩 샘플 미리보기)
def dp(df,row=2,col=15):
    n=0
    while 1:
        if n*col >= df.shape[1]:
            break
        n += 1
        display(df.iloc[:row,col*(n-1):col*n])
        

### 세로로 길때 전부 살펴보기
def sero(df,N):
    K = (len(df)//N)+2
    for i in range(K):
        display(df.iloc[N*i:N*(i+1),:])



def pre(df):
    # 1) numeric --> int32 전환 (±21억 이내)
    cols = ["customer_id", "pre_discount", "post_discount", "cashback_amount", "discount_amount", "total_promotion"]  # 6개 피쳐
    df[cols] = df[cols].astype(np.int32)

    # 2) n_unique: categorical feature --> int8 전환
    for col in ["push_permission", "is_foreigner"]:
        df[col] = df[col].apply(lambda x: 1 if x==True else 0)  ## (푸쉬승인:1, 외국인:1)

    df["gender"] = df["gender"].apply(lambda x: 1 if x=="male" else 0)  ## (남성:1)

    cols = ["push_permission", "gender", "is_foreigner", "merchant_id"]  # 4개 피쳐
    df[cols] = df[cols].astype(np.int8)

    # 3) time_series --> 시간단위까지는 안 쓸래 --> datetime64 변환
    for col in ["created_at", "sign_up_date"]:
        df[col] = df[col].apply(lambda x: x[:11])

    cols = ["created_at", "sign_up_date", 'birthday']  # 3개 피쳐
    df[cols] = df[cols].astype("datetime64")
    return df



def pre_buy(df):
    # 1) 컬럼명 간소화
    df.columns=["id", "buy", "before", "after", "back", "discnt", "total", "push", "male", "foreigner", "birth", "sign_up", "merchant"]

    # 2) 구매일자 년/월/일/요일 추출
    df["buy_y"] = df["buy"].dt.year.astype("int16")
    df["buy_m"] = df["buy"].dt.month.astype("int8")
    df["buy_d"] = df["buy"].dt.day.astype("int8")
    df["buy_7"] = df["buy"].dt.day_name()
    to_7 = {"Sunday":0, "Monday":1, "Tuesday":2, "Wednesday":3, "Thursday":4, "Friday":5, "Saturday":6}
    df["buy_7"] = df["buy_7"].apply(lambda x: to_7[x]).astype("int8")

    # 3) 가입일자 년/월/일 추출
    df["sign_y"] = df["sign_up"].dt.year.astype("int16")
    df["sign_m"] = df["sign_up"].dt.month.astype("int8")
    df["sign_d"] = df["sign_up"].dt.day.astype("int8")
        
    # 4) (구매 당시) 나이 feature 추가
    df["age"] = df["buy_y"] - df["birth"].dt.year + 1
    df["age"] = df["age"].astype("int8")

    # 5) 거래건별: 상품가 대비 페이백비율, 할인비율, 총혜택비율 feature 추가
    df["b_rate"] = round(  100*df.back/df.before,   2  )
    df["d_rate"] = round(  100*df.discnt/df.before, 2  )
    df["t_rate"] = round(  100*df.total/df.before,  2  )

    # 6) 구매건수 카운트 추가 --> (groupby)
    df["cnt_b"] = 1
    df["cnt_b"] = df["cnt_b"].astype(np.int16)
    return df



def pre_personal(df):
    # 1) merchant 원핫 인코딩
    df2 = pd.get_dummies(  data=df, columns=["merchant"]  )
    dic = {f"merchant_{i}":f"mc{i}" for i in range(1,11)}  ## 이름 짧게 바꾸기
    df2 = df2.rename(  columns=dic  )
    df2.loc[:,"mc1":"mc10"] = df2.loc[:,"mc1":"mc10"].astype(np.int16)  ## dtype 바꾸기

    ### groupby().sum() 유용한 feature 생성 --> df2
    df2 = df2[['id', 'before', 'after', 'back', 'discnt', 'total', 'cnt_b', 'mc1', 'mc2', 'mc3', 'mc4', 'mc5', 'mc6', 'mc7', 'mc8', 'mc9', 'mc10']]
    df2 = df2.groupby(by="id").sum()
    df2.cnt_b = df2.cnt_b.astype(np.int16)

    ### groupby().min() 유용한 feature 생성 --> df3
    df3 = df[['id', 'push', 'male', 'foreigner', 'birth', 'sign_up', 'sign_y', 'sign_m', 'sign_d', 'age']]  ## age: 이 때의 나이는 맨 마지막 구매 시점
    df3 = df3.groupby(by="id").max()

    ### df2 + df3 = merge
    df2 = pd.merge( df2, df3, how="left", on=["id"] )
    df2 = df2.reset_index()
    df2 = df2[['id', 'male', 'age', 'push', 'foreigner', 'cnt_b', 'before', 'after', 'back', 'discnt', 'total', 'birth', 'sign_up', 'sign_y', 'sign_m', 'sign_d',
            'mc1', 'mc2', 'mc3', 'mc4', 'mc5', 'mc6', 'mc7', 'mc8', 'mc9', 'mc10']]

    ### cnt_p: 인원별 카운트 feature 생성
    df2["cnt_p"] = 1
    df2["cnt_p"] = df2["cnt_p"].astype(np.int16)

    ### 고객별: 상품가 대비 페이백비율, 할인비율, 총혜택비율 feature 추가
    df2["b_rate"] = round(  100*df2.back/df2.before,   2  )
    df2["d_rate"] = round(  100*df2.discnt/df2.before, 2  )
    df2["t_rate"] = round(  100*df2.total/df2.before,  2  )
    return df2



### infox: full data
def infox(df):
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))
    print( f"◆◆◆ {df.shape}: Total shape ◆◆◆" )
    df_num = df.select_dtypes( include='number', exclude=['cfloat','complex64',"complex128"] )
    if df_num.shape[1] > 0:
        print( f"---{df_num.shape}: Numeric Data: only Real Number ↓↓↓ " + "-"*27 )
        df_info = pd.DataFrame(  [[ i for i in range(df_num.shape[1]) ]], columns=df_num.columns, index=["NO"]  )
        df_info.loc["Column"] = df_num.columns
        df_info.loc["null"] = len(df_num) - df_num.count()
        df_info.loc["null(%)"] = round( 100 * (len(df_num)-df_num.count()) / len(df_num), 1)
        df_info.loc["dtype"] = df_num.dtypes
        df_info.loc["n_uniq"] = df_num.nunique()
        df_info.loc["|"] = "|"
        df_info.loc["Mean"] = df_num.mean()
        df_info.loc["Std"] = df_num.std(ddof=0)
        df_info.loc["|max-min|"] = df_num.max() - df_num.min()
        df_info.loc["│"] = "│"
        df_info.loc["min"] = df_num.min()
        df_info.loc["Q1"] = df_num.quantile(0.25)
        df_info.loc["median"] = df_num.median()   
        df_info.loc["Q3"] = df_num.quantile(0.75)
        df_info.loc["max"] = df_num.max() 
        df_info = df_info.T
        df_info["NO"] = df_info["NO"].astype(int)
        df_info["null"] = df_info["null"].astype(int)
        for i in range(len(df_info)):
            for j in [2,5,7,8,9]+list(range(11,df_info.shape[1])):
                if df_info.iloc[i,j]==int(df_info.iloc[i,j]):
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,}').split(".")[0] 
                elif abs( df_info.iloc[i,j] ) >= 1000:
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,}').split(".")[0] 
                elif abs( df_info.iloc[i,j] ) >= 100:
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,.1f}').rstrip("0") 
                elif abs( df_info.iloc[i,j] ) >= 10:
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,.2f}').rstrip("0") 
                elif abs( df_info.iloc[i,j] ) >= 1:
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,.3f}').rstrip("0")
                else:
                    df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,.4f}').rstrip("0")
        df_info = df_info.astype(str)
        df_col = pd.DataFrame( [df_info.columns], columns=df_info.columns )
        df_info = pd.concat( [df_col,df_info] )
        df_len = df_info.copy()
        for i in range(len(df_len)):
            for j in range(df_len.shape[1]):
                df_len.iloc[i,j]=len(df_len.iloc[i,j])
        for i in range(len(df_info)):
            for j in range(df_info.shape[1]):
                print(df_info.iloc[i,j].rjust(  df_len.max()[j]  ), end="  ")
            print()
    df_time = df.select_dtypes(include='datetime')
    if df_time.shape[1] > 0:
        print( f"---{df_time.shape}: DateTime Data ↓↓↓ " + "-"*44 )
        df_info = pd.DataFrame(  [[ i for i in range(df_time.shape[1]) ]], columns=df_time.columns, index=["NO"]  )
        df_info.loc["Column"] = df_time.columns  
        df_info.loc["null"] = len(df_time) - df_time.count()
        df_info.loc["null(%)"] = round( 100 * (len(df_time)-df_time.count()) / len(df_time), 1)
        df_info.loc["dtype"] = df_time.dtypes
        df_info.loc["n_uniq"] = df_time.nunique()
        df_info.loc["|"] = "|"
        df_info.loc["min"] = df_time.min()
        df_info.loc["max"] = df_time.max() 
        df_info = df_info.T
        df_info["NO"] = df_info["NO"].astype(int) 
        df_info["null"] = df_info["null"].astype(int)
        for i in range(len(df_info)):
            for j in [2,5]:
                df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,}').split(".")[0]
        df_info = df_info.astype(str)
        df_col = pd.DataFrame( [df_info.columns], columns=df_info.columns )
        df_info = pd.concat( [df_col,df_info] )
        df_len = df_info.copy()
        for i in range(len(df_len)):
            for j in range(df_len.shape[1]):
                df_len.iloc[i,j]=len(df_len.iloc[i,j])
        for i in range(len(df_info)):
            for j in range(df_info.shape[1]):
                print(df_info.iloc[i,j].rjust(  df_len.max()[j]  ), end="  ") 
            print()
    df_obj_1 = df.select_dtypes( exclude=['number','datetime'] )
    df_obj_2 = df.select_dtypes( include=['cfloat','complex64',"complex128"] )
    df_obj = pd.concat( [df_obj_1,df_obj_2], axis=1 )
    if df_obj.shape[1] > 0:
        print( f"---{df_obj.shape}: etc Data: Object, Complex Numbers, ... ↓↓↓ " + "-"*19 )
        df_info = pd.DataFrame(  [[ i for i in range(df_obj.shape[1]) ]], columns=df_obj.columns, index=["NO"]  )
        df_info.loc["Column"] = df_obj.columns 
        df_info.loc["null"] = len(df_obj) - df_obj.count()
        df_info.loc["null(%)"] = round( 100 * (len(df_obj)-df_obj.count()) / len(df_obj), 1)
        df_info.loc["dtype"] = df_obj.dtypes
        df_info.loc["n_uniq"] = df_obj.nunique()
        df_info.loc["|"] = "|"
        freq_list = []
        for i in range(df_obj.shape[1]):
            freq_list.append(  list(df_obj.iloc[:,i].value_counts())[0]  )
        df_info.loc["No1_freq_count"] = freq_list
        df_info.loc["rate(%)"] = [ f"{100*x/len(df_obj):.1f}" for x in freq_list ]
        df_info.loc["value"] = list(  df_obj.mode().iloc[0]  )
        df_info = df_info.T
        df_info["NO"] = df_info["NO"].astype(int) 
        df_info["null"] = df_info["null"].astype(int)
        for i in range(len(df_info)):
            for j in [2,5,7]:
                df_info.iloc[i,j]=str(f'{df_info.iloc[i,j]:,}').split(".")[0]
        df_info = df_info.astype(str)
        df_col = pd.DataFrame( [df_info.columns], columns=df_info.columns )
        df_info = pd.concat( [df_col,df_info] )
        df_len = df_info.copy()
        for i in range(len(df_len)):
            for j in range(df_len.shape[1]):
                df_len.iloc[i,j]=len(df_len.iloc[i,j])
        for i in range(len(df_info)):
            for j in range(df_info.shape[1]):
                if j !=9:
                    print(df_info.iloc[i,j].rjust(  df_len.max()[j]  ), end="  ")
                else:
                    print(df_info.iloc[i,j].ljust(  df_len.max()[j]  ), end="  ")
            print()
    print("-"*78,"\n")

