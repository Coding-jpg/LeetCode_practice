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
    