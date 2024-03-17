'''
595
'''
import pandas as pd

def big_countries(world: pd.DataFrame) -> pd.DataFrame:
    print(world.head())
    return world[(world['area'] >= 3000000) | (world['population'] >= 25000000)][['name','population','area']]

'''
1757
'''
import pandas as pd

def find_products(products: pd.DataFrame) -> pd.DataFrame:
    return products.loc[(products['low_fats'] == 'Y') & (products['recyclable'] == 'Y'), ['product_id']]
    
'''
183
'''
import pandas as pd

def find_customers(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = customers[~customers['id'].isin(orders['customerId'])]
    df = df[['name']].rename(columns={'name': 'Customers'})
    return df    

'''
1683
'''
import pandas as pd

def invalid_tweets(tweets: pd.DataFrame) -> pd.DataFrame:
    df = tweets[tweets['content'].str.len() > 15]
    df = df[['tweet_id']]
    return df

'''
1873
'''
import pandas as pd

def calculate_special_bonus(employees: pd.DataFrame) -> pd.DataFrame:
    condition = ~(((employees['employee_id'] % 2) != 0) & (employees['name'].str[0] != 'M'))
    employees.loc[condition, 'salary'] = 0
    employees = employees[['employee_id', 'salary']].rename(columns={'salary': 'bonus'}).sort_values(by='employee_id')
    return employees

'''
1667
'''
import pandas as pd

def fix_names(users: pd.DataFrame) -> pd.DataFrame:
    users['name'] = users['name'].str.capitalize()
    users = users.sort_values(by='user_id')
    return users

'''
1517
'''
import pandas as pd

def valid_emails(users: pd.DataFrame) -> pd.DataFrame:
    df = users[users['mail'].str.match(r"^[a-zA-Z][a-zA-Z0-9_.-]*\@leetcode\.com$")]
    return df

'''
1527
'''
import pandas as pd

def find_patients(patients: pd.DataFrame) -> pd.DataFrame:
    return patients[patients['conditions'].str.contains(r"(^|\s)DIAB1")]

'''
177
'''
import pandas as pd

def nth_highest_salary(employee: pd.DataFrame, N: int) -> pd.DataFrame:
    sorted_df = employee.sort_values(by='salary', ascending=False).drop_duplicates(subset='salary')

    if 0 < N <= len(sorted_df):
        nth_salary = sorted_df.iloc[N-1]['salary']
    else:
        nth_salary = None 

    res_df = pd.DataFrame({
        f'getNthHighestSalary({N})': [nth_salary]
    })
    return res_df

'''
176
'''
import pandas as pd

def second_highest_salary(employee: pd.DataFrame) -> pd.DataFrame:
    sorted_df = employee.sort_values(by='salary', ascending=False).drop_duplicates(subset='salary')

    if 2 <= len(sorted_df):
        sth_salary = sorted_df.iloc[1]['salary']
    else:
        sth_salary = None 

    res_df = pd.DataFrame({
        f'SecondHighestSalary': [sth_salary]
    })
    return res_df

'''
184
'''
import pandas as pd

def department_highest_salary(employee: pd.DataFrame, department: pd.DataFrame) -> pd.DataFrame:
    def get_top_earners(group):
        top_salary = group['Salary'].max()
        return group[group['Salary'] == top_salary]

    df = pd.merge(employee, department, left_on='departmentId', right_on='id')
    df = df.drop(['id_x', 'id_y', 'departmentId'], axis=1) \
            .rename(columns={'name_x':'Employee', 'name_y':'Department', 'salary':'Salary'})
    df = df[['Department', 'Employee', 'Salary']] \
            .sort_values(by='Salary', ascending=False) \
            .groupby('Department').apply(get_top_earners)
    return df
    
'''
178
'''
import pandas as pd

def order_scores(scores: pd.DataFrame) -> pd.DataFrame:
    scores['rank'] = scores['score'].rank(ascending=False, method='dense').astype(int)
    return scores.sort_values(by='score', ascending=False).drop('id', axis=1)

'''
196
'''
import pandas as pd

def delete_duplicate_emails(person: pd.DataFrame) -> None:

    # 使用 groupby 按 'email' 分组，并使用 'agg' 来找到每组 'id' 的最小值
    result = person.groupby('email').agg(min_id=('id', 'min'))
    # 确定要保留的行
    keep_rows = person['id'].isin(result['min_id'])
    # 使用 .loc 更新原始 DataFrame
    person.loc[~keep_rows, :] = None
    # 删除值为 None 的行
    person.dropna(inplace=True)

'''
1795
'''
import pandas as pd

def rearrange_products_table(products: pd.DataFrame) -> pd.DataFrame:
    df = pd.melt(
        products,
        id_vars='product_id',
        value_vars=['store1','store2','store3'],
        var_name='store',
        value_name='price'
    ).dropna()
    return df
    
'''
1907
'''
import pandas as pd

def count_salary_categories(accounts: pd.DataFrame) -> pd.DataFrame:
    low_count = (accounts['income'] < 20000).sum()
    average_count = ((accounts['income'] >= 20000) & (accounts['income'] <= 50000)).sum()
    high_count = (accounts['income'] > 50000).sum()

    ans = pd.DataFrame({
        'category': ['High Salary','Low Salary', 'Average Salary'],
        'accounts_count': [high_count, low_count, average_count]
    })
    return ans

'''
1741
'''
import pandas as pd

def total_time(employees: pd.DataFrame) -> pd.DataFrame:
    employees['total_time'] = employees['out_time'] - employees['in_time']
    employees = employees.groupby(['emp_id', 'event_day'])['total_time'].sum().reset_index()
    employees = employees[['event_day', 'emp_id', 'total_time']]
    employees = employees.rename(columns={'event_day':'day'})
    return employees

'''
511
'''
import pandas as pd

def game_analysis(activity: pd.DataFrame) -> pd.DataFrame:
    first_login = activity.groupby('player_id')['event_date'].min().reset_index()
    first_login = first_login.rename(columns={'event_date':'first_login'})
    return first_login

'''
2356
'''
import pandas as pd

def count_unique_subjects(teacher: pd.DataFrame) -> pd.DataFrame:
    df = teacher.groupby('teacher_id')['subject_id'].nunique().reset_index()
    df = df.rename(columns={'subject_id': 'cnt'})
    return df

'''
596
'''
import pandas as pd

def find_classes(courses: pd.DataFrame) -> pd.DataFrame:
    class_counts = courses.groupby('class')['class'].transform('count')
    df = courses[class_counts >= 5]['class'].drop_duplicates().reset_index(drop=True).to_frame()
    return df

'''
586
'''
import pandas as pd

def largest_orders(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.isna().all().all(): return orders['customer_number'].to_frame()
    df = orders.groupby('customer_number').count().reset_index()
    print(df)
    df = df[df['order_number'] == df['order_number'].max()]['customer_number'].to_frame()
    return df

'''
1484
'''
import pandas as pd

def categorize_products(activities: pd.DataFrame) -> pd.DataFrame:
    df = activities.groupby('sell_date', as_index=False)['product'].agg(num_sold='nunique', products=lambda x: ','.join(sorted(x.unique())))
    return df

'''
1693
'''
import pandas as pd

def daily_leads_and_partners(daily_sales: pd.DataFrame) -> pd.DataFrame:
    df = daily_sales.groupby(['date_id', 'make_name'], as_index=False).agg({'lead_id': 'nunique', 'partner_id': 'nunique'}).rename(columns={'lead_id':'unique_leads', 'partner_id':'unique_partners'})
    return df

'''
1050
'''
import pandas as pd

def actors_and_directors(actor_director: pd.DataFrame) -> pd.DataFrame:
    df = actor_director.groupby(['actor_id', 'director_id'], as_index=False).agg('count').rename(columns={'timestamp':'count'})
    return df[df['count'] >= 3][['actor_id', 'director_id']]

'''
1378
'''
import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(employees, employee_uni, left_on='id', right_on='id', how='left')
    df = df[['unique_id', 'name']]
    return df

'''
1280
'''
import pandas as pd

def students_and_examinations(students: pd.DataFrame, subjects: pd.DataFrame, examinations: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(
            pd.merge(students,subjects,how='cross'),
            examinations.groupby(['subject_name','student_id']).size().reset_index(name='attended_exams'),
            on = ['student_id','subject_name'],
            how = 'left'
        ).sort_values(by= ['student_id','subject_name'])\
        .reindex(['student_id',	'student_name',	'subject_name',	'attended_exams'],axis=1)
    df['attended_exams'] = df['attended_exams'].fillna(0)
    return df

'''
607
'''
import pandas as pd

def sales_person(sales_person: pd.DataFrame, company: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(orders, company, on='com_id')

    red_orders = df[df['name'] == 'RED']

    invalid_ids = red_orders.sales_id.unique()

    valid_sales_person = sales_person[~sales_person['sales_id'].isin(invalid_ids)]    

    return valid_sales_person[['name']]

'''
570
'''
import pandas as pd

def find_managers(employee: pd.DataFrame) -> pd.DataFrame:
    manager_counts = employee.groupby('managerId').size()
    managers_with_5_or_more = manager_counts[manager_counts >= 5].index
    managers_names = employee[employee['id'].isin(managers_with_5_or_more)]['name']
    return managers_names.to_frame()