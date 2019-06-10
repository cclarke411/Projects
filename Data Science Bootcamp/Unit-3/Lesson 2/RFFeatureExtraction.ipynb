{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\IPython\\core\\interactiveshell.py:3020: DtypeWarning: Columns (0,19,55) have mixed types. Specify dtype option on import or set low_memory=False.\n",
      "  interactivity=interactivity, compiler=compiler, result=result)\n"
     ]
    }
   ],
   "source": [
    "# Replace the path with the correct path for your data.\n",
    "y2015 = pd.read_csv(\n",
    "    'https://www.dropbox.com/s/0so14yudedjmm5m/LoanStats3d.csv?dl=1',\n",
    "    skipinitialspace=True,\n",
    "    header=1\n",
    ")\n",
    "\n",
    "# Note the warning about dtypes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(421097, 111)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y2015.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id\n",
      "421097\n",
      "term\n",
      "2\n",
      "int_rate\n",
      "110\n",
      "grade\n",
      "7\n",
      "sub_grade\n",
      "35\n",
      "emp_title\n",
      "120812\n",
      "emp_length\n",
      "11\n",
      "home_ownership\n",
      "4\n",
      "verification_status\n",
      "3\n",
      "issue_d\n",
      "12\n",
      "loan_status\n",
      "7\n",
      "pymnt_plan\n",
      "1\n",
      "url\n",
      "421095\n",
      "desc\n",
      "34\n",
      "purpose\n",
      "14\n",
      "title\n",
      "27\n",
      "zip_code\n",
      "914\n",
      "addr_state\n",
      "49\n",
      "earliest_cr_line\n",
      "668\n",
      "revol_util\n",
      "1211\n",
      "initial_list_status\n",
      "2\n",
      "last_pymnt_d\n",
      "25\n",
      "next_pymnt_d\n",
      "4\n",
      "last_credit_pull_d\n",
      "26\n",
      "application_type\n",
      "2\n",
      "verification_status_joint\n",
      "3\n"
     ]
    }
   ],
   "source": [
    "categorical = y2015.select_dtypes(include=['object'])\n",
    "for i in categorical:\n",
    "    column = categorical[i]\n",
    "    print(i)\n",
    "    print(column.nunique())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    }
   },
   "outputs": [],
   "source": [
    "# Convert ID and Interest Rate to numeric.\n",
    "y2015['id'] = pd.to_numeric(y2015['id'], errors='coerce')\n",
    "y2015['int_rate'] = pd.to_numeric(y2015['int_rate'].str.strip('%'), errors='coerce')\n",
    "\n",
    "# Drop other columns with many unique variables\n",
    "y2015.drop(['url', 'emp_title', 'zip_code', 'earliest_cr_line', 'revol_util',\n",
    "            'sub_grade', 'addr_state', 'desc'], 1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>member_id</th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>funded_amnt</th>\n",
       "      <th>funded_amnt_inv</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>sub_grade</th>\n",
       "      <th>...</th>\n",
       "      <th>num_tl_90g_dpd_24m</th>\n",
       "      <th>num_tl_op_past_12m</th>\n",
       "      <th>pct_tl_nvr_dlq</th>\n",
       "      <th>percent_bc_gt_75</th>\n",
       "      <th>pub_rec_bankruptcies</th>\n",
       "      <th>tax_liens</th>\n",
       "      <th>tot_hi_cred_lim</th>\n",
       "      <th>total_bal_ex_mort</th>\n",
       "      <th>total_bc_limit</th>\n",
       "      <th>total_il_high_credit_limit</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>421092</th>\n",
       "      <td>36271333</td>\n",
       "      <td>38982739.0</td>\n",
       "      <td>13000.0</td>\n",
       "      <td>13000.0</td>\n",
       "      <td>13000.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>15.99%</td>\n",
       "      <td>316.07</td>\n",
       "      <td>D</td>\n",
       "      <td>D2</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>51239.0</td>\n",
       "      <td>34178.0</td>\n",
       "      <td>10600.0</td>\n",
       "      <td>33239.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421093</th>\n",
       "      <td>36490806</td>\n",
       "      <td>39222577.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>12000.0</td>\n",
       "      <td>60 months</td>\n",
       "      <td>19.99%</td>\n",
       "      <td>317.86</td>\n",
       "      <td>E</td>\n",
       "      <td>E3</td>\n",
       "      <td>...</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>66.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>96919.0</td>\n",
       "      <td>58418.0</td>\n",
       "      <td>9700.0</td>\n",
       "      <td>69919.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421094</th>\n",
       "      <td>36271262</td>\n",
       "      <td>38982659.0</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>11.99%</td>\n",
       "      <td>664.20</td>\n",
       "      <td>B</td>\n",
       "      <td>B5</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>100.0</td>\n",
       "      <td>50.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>43740.0</td>\n",
       "      <td>33307.0</td>\n",
       "      <td>41700.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421095</th>\n",
       "      <td>Total amount funded in policy code 1: 6417608175</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421096</th>\n",
       "      <td>Total amount funded in policy code 2: 1944088810</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 111 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                      id   member_id  \\\n",
       "421092                                          36271333  38982739.0   \n",
       "421093                                          36490806  39222577.0   \n",
       "421094                                          36271262  38982659.0   \n",
       "421095  Total amount funded in policy code 1: 6417608175         NaN   \n",
       "421096  Total amount funded in policy code 2: 1944088810         NaN   \n",
       "\n",
       "        loan_amnt  funded_amnt  funded_amnt_inv        term int_rate  \\\n",
       "421092    13000.0      13000.0          13000.0   60 months   15.99%   \n",
       "421093    12000.0      12000.0          12000.0   60 months   19.99%   \n",
       "421094    20000.0      20000.0          20000.0   36 months   11.99%   \n",
       "421095        NaN          NaN              NaN         NaN      NaN   \n",
       "421096        NaN          NaN              NaN         NaN      NaN   \n",
       "\n",
       "        installment grade sub_grade            ...              \\\n",
       "421092       316.07     D        D2            ...               \n",
       "421093       317.86     E        E3            ...               \n",
       "421094       664.20     B        B5            ...               \n",
       "421095          NaN   NaN       NaN            ...               \n",
       "421096          NaN   NaN       NaN            ...               \n",
       "\n",
       "       num_tl_90g_dpd_24m num_tl_op_past_12m pct_tl_nvr_dlq  percent_bc_gt_75  \\\n",
       "421092                0.0                3.0          100.0              50.0   \n",
       "421093                1.0                2.0           95.0              66.7   \n",
       "421094                0.0                1.0          100.0              50.0   \n",
       "421095                NaN                NaN            NaN               NaN   \n",
       "421096                NaN                NaN            NaN               NaN   \n",
       "\n",
       "       pub_rec_bankruptcies tax_liens tot_hi_cred_lim total_bal_ex_mort  \\\n",
       "421092                  1.0       0.0         51239.0           34178.0   \n",
       "421093                  0.0       0.0         96919.0           58418.0   \n",
       "421094                  0.0       1.0         43740.0           33307.0   \n",
       "421095                  NaN       NaN             NaN               NaN   \n",
       "421096                  NaN       NaN             NaN               NaN   \n",
       "\n",
       "       total_bc_limit total_il_high_credit_limit  \n",
       "421092        10600.0                    33239.0  \n",
       "421093         9700.0                    69919.0  \n",
       "421094        41700.0                        0.0  \n",
       "421095            NaN                        NaN  \n",
       "421096            NaN                        NaN  \n",
       "\n",
       "[5 rows x 111 columns]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y2015.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "y2015 = y2015[:-2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    }
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>member_id</th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>funded_amnt</th>\n",
       "      <th>funded_amnt_inv</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>annual_inc</th>\n",
       "      <th>dti</th>\n",
       "      <th>delinq_2yrs</th>\n",
       "      <th>...</th>\n",
       "      <th>last_credit_pull_d_Nov-2016</th>\n",
       "      <th>last_credit_pull_d_Oct-2015</th>\n",
       "      <th>last_credit_pull_d_Oct-2016</th>\n",
       "      <th>last_credit_pull_d_Sep-2015</th>\n",
       "      <th>last_credit_pull_d_Sep-2016</th>\n",
       "      <th>application_type_INDIVIDUAL</th>\n",
       "      <th>application_type_JOINT</th>\n",
       "      <th>verification_status_joint_Not Verified</th>\n",
       "      <th>verification_status_joint_Source Verified</th>\n",
       "      <th>verification_status_joint_Verified</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>68009401</td>\n",
       "      <td>72868139.0</td>\n",
       "      <td>16000.0</td>\n",
       "      <td>16000.0</td>\n",
       "      <td>16000.0</td>\n",
       "      <td>14.85</td>\n",
       "      <td>379.39</td>\n",
       "      <td>48000.0</td>\n",
       "      <td>33.18</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>68354783</td>\n",
       "      <td>73244544.0</td>\n",
       "      <td>9600.0</td>\n",
       "      <td>9600.0</td>\n",
       "      <td>9600.0</td>\n",
       "      <td>7.49</td>\n",
       "      <td>298.58</td>\n",
       "      <td>60000.0</td>\n",
       "      <td>22.44</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>68466916</td>\n",
       "      <td>73356753.0</td>\n",
       "      <td>25000.0</td>\n",
       "      <td>25000.0</td>\n",
       "      <td>25000.0</td>\n",
       "      <td>7.49</td>\n",
       "      <td>777.55</td>\n",
       "      <td>109000.0</td>\n",
       "      <td>26.02</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>68466961</td>\n",
       "      <td>73356799.0</td>\n",
       "      <td>28000.0</td>\n",
       "      <td>28000.0</td>\n",
       "      <td>28000.0</td>\n",
       "      <td>6.49</td>\n",
       "      <td>858.05</td>\n",
       "      <td>92000.0</td>\n",
       "      <td>21.60</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>68495092</td>\n",
       "      <td>73384866.0</td>\n",
       "      <td>8650.0</td>\n",
       "      <td>8650.0</td>\n",
       "      <td>8650.0</td>\n",
       "      <td>19.89</td>\n",
       "      <td>320.99</td>\n",
       "      <td>55000.0</td>\n",
       "      <td>25.49</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 237 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         id   member_id  loan_amnt  funded_amnt  funded_amnt_inv  int_rate  \\\n",
       "0  68009401  72868139.0    16000.0      16000.0          16000.0     14.85   \n",
       "1  68354783  73244544.0     9600.0       9600.0           9600.0      7.49   \n",
       "2  68466916  73356753.0    25000.0      25000.0          25000.0      7.49   \n",
       "3  68466961  73356799.0    28000.0      28000.0          28000.0      6.49   \n",
       "4  68495092  73384866.0     8650.0       8650.0           8650.0     19.89   \n",
       "\n",
       "   installment  annual_inc    dti  delinq_2yrs  \\\n",
       "0       379.39     48000.0  33.18          0.0   \n",
       "1       298.58     60000.0  22.44          0.0   \n",
       "2       777.55    109000.0  26.02          0.0   \n",
       "3       858.05     92000.0  21.60          0.0   \n",
       "4       320.99     55000.0  25.49          0.0   \n",
       "\n",
       "                  ...                  last_credit_pull_d_Nov-2016  \\\n",
       "0                 ...                                            0   \n",
       "1                 ...                                            0   \n",
       "2                 ...                                            0   \n",
       "3                 ...                                            0   \n",
       "4                 ...                                            0   \n",
       "\n",
       "   last_credit_pull_d_Oct-2015  last_credit_pull_d_Oct-2016  \\\n",
       "0                            0                            0   \n",
       "1                            0                            0   \n",
       "2                            0                            0   \n",
       "3                            0                            0   \n",
       "4                            0                            0   \n",
       "\n",
       "   last_credit_pull_d_Sep-2015  last_credit_pull_d_Sep-2016  \\\n",
       "0                            0                            0   \n",
       "1                            0                            0   \n",
       "2                            0                            0   \n",
       "3                            0                            0   \n",
       "4                            0                            0   \n",
       "\n",
       "   application_type_INDIVIDUAL  application_type_JOINT  \\\n",
       "0                            1                       0   \n",
       "1                            1                       0   \n",
       "2                            1                       0   \n",
       "3                            1                       0   \n",
       "4                            1                       0   \n",
       "\n",
       "   verification_status_joint_Not Verified  \\\n",
       "0                                       0   \n",
       "1                                       0   \n",
       "2                                       0   \n",
       "3                                       0   \n",
       "4                                       0   \n",
       "\n",
       "   verification_status_joint_Source Verified  \\\n",
       "0                                          0   \n",
       "1                                          0   \n",
       "2                                          0   \n",
       "3                                          0   \n",
       "4                                          0   \n",
       "\n",
       "   verification_status_joint_Verified  \n",
       "0                                   0  \n",
       "1                                   0  \n",
       "2                                   0  \n",
       "3                                   0  \n",
       "4                                   0  \n",
       "\n",
       "[5 rows x 237 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "XX =pd.get_dummies(y2015)\n",
    "XX.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    },
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "C:\\Users\\clyde\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    }
   ],
   "source": [
    "from sklearn import ensemble\n",
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "rfc = ensemble.RandomForestClassifier()\n",
    "X = y2015.drop('loan_status', 1)\n",
    "Y = y2015['loan_status']\n",
    "X = pd.get_dummies(X)\n",
    "X = X.dropna(axis=1)\n",
    "\n",
    "cross_val_score(rfc, X, Y, cv=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "run_control": {
     "frozen": false,
     "read_only": false
    }
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/sklearn/preprocessing/data.py:625: DataConversionWarning: Data with input dtype uint8, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.partial_fit(X, y)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/base.py:462: DataConversionWarning: Data with input dtype uint8, float64 were all converted to float64 by StandardScaler.\n",
      "  return self.fit(X, **fit_params).transform(X)\n"
     ]
    }
   ],
   "source": [
    "import math\n",
    "from matplotlib.mlab import PCA as mlabPCA\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.decomposition import PCA \n",
    "\n",
    "X = StandardScaler().fit_transform(X)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(201,)"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Xt = X.T\n",
    "Cx = np.cov(Xt)\n",
    "eig_val_cov, eig_vec_cov = np.linalg.eig(Cx)\n",
    "eig_val_cov.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([10.74242742+0.j,  6.17082352+0.j,  5.9720621 +0.j,  4.7379119 +0.j,\n",
       "        3.77564364+0.j,  3.21357484+0.j,  3.14702889+0.j,  2.9237168 +0.j,\n",
       "        2.62559544+0.j,  2.56554441+0.j])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "eig_val_cov[0:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/numpy/core/numeric.py:538: ComplexWarning: Casting complex values to real discards the imaginary part\n",
      "  return array(a, dtype, copy=False, order=order)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x7fcfbf9fb588>]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(eig_val_cov)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "87"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(eig_val_cov[eig_val_cov>1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import PCA as sklearnPCA\n",
    "sklearn_pca = sklearnPCA(n_components=87)\n",
    "X_sklearn = sklearn_pca.fit_transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(421095, 87)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_sklearn.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n",
      "/opt/conda/lib/python3.7/site-packages/sklearn/ensemble/forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([0.94341146, 0.9457149 , 0.94849327, 0.95234024, 0.9386369 ,\n",
       "       0.94875327, 0.9448099 , 0.95067563, 0.94100743, 0.94433097])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rfc = ensemble.RandomForestClassifier()\n",
    "Y = y2015['loan_status']\n",
    "\n",
    "cross_val_score(rfc, X_sklearn, Y, cv=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1.48578579,  1.50710837,  0.08863448, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839],\n",
       "       [ 1.52223613,  1.54425274, -0.65804201, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839],\n",
       "       [ 1.53407024,  1.55532575,  1.13864831, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839],\n",
       "       ...,\n",
       "       [-1.86373287, -1.83676857, -0.26137012, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839],\n",
       "       [-1.8405705 , -1.81310089, -0.37803833, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839],\n",
       "       [-1.86374037, -1.83677647,  0.55530729, ..., -0.02593279,\n",
       "        -0.01203667, -0.01991839]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "hide_input": false,
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  },
  "toc": {
   "colors": {
    "hover_highlight": "#DAA520",
    "running_highlight": "#FF0000",
    "selected_highlight": "#FFD700"
   },
   "moveMenuLeft": true,
   "nav_menu": {
    "height": "86px",
    "width": "252px"
   },
   "navigate_menu": true,
   "number_sections": true,
   "sideBar": true,
   "threshold": 4,
   "toc_cell": false,
   "toc_section_display": "block",
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}


