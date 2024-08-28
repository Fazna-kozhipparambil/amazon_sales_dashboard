{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNnInL6I8oaNO/KQoEFoE2E"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        " 1.Load and Explore the Data"
      ],
      "metadata": {
        "id": "u0sAbwaw4hVG"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "SMhdd2rY5COR",
        "outputId": "5f001b02-d9d8-4db2-80eb-10df1d3562ee"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.38.0-py2.py3-none-any.whl.metadata (8.5 kB)\n",
            "Requirement already satisfied: altair<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Requirement already satisfied: blinker<2,>=1.0.0 in /usr/lib/python3/dist-packages (from streamlit) (1.4)\n",
            "Requirement already satisfied: cachetools<6,>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.7)\n",
            "Requirement already satisfied: numpy<3,>=1.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<25,>=20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (24.1)\n",
            "Requirement already satisfied: pandas<3,>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.1.4)\n",
            "Requirement already satisfied: pillow<11,>=7.1.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.4.0)\n",
            "Requirement already satisfied: protobuf<6,>=3.20 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (14.0.2)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.32.3)\n",
            "Requirement already satisfied: rich<14,>=10.14.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.8.0)\n",
            "Collecting tenacity<9,>=8.1.0 (from streamlit)\n",
            "  Downloading tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.3.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.12.2)\n",
            "Collecting gitpython!=3.1.19,<4,>=3.0.7 (from streamlit)\n",
            "  Downloading GitPython-3.1.43-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado<7,>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.3.3)\n",
            "Collecting watchdog<5,>=2.1.5 (from streamlit)\n",
            "  Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl.metadata (38 kB)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<6,>=4.0->streamlit) (0.12.1)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading gitdb-4.0.11-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=1.3.0->streamlit) (2024.1)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (3.8)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2.0.7)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.27->streamlit) (2024.7.4)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich<14,>=10.14.0->streamlit) (2.16.1)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit)\n",
            "  Downloading smmap-5.0.1-py3-none-any.whl.metadata (4.3 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.5)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.2.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.12.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.35.1)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.20.0)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.3.0->streamlit) (1.16.0)\n",
            "Downloading streamlit-1.38.0-py2.py3-none-any.whl (8.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.7/8.7 MB\u001b[0m \u001b[31m43.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading GitPython-3.1.43-py3-none-any.whl (207 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m55.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading tenacity-8.5.0-py3-none-any.whl (28 kB)\n",
            "Downloading watchdog-4.0.2-py3-none-manylinux2014_x86_64.whl (82 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.9/82.9 kB\u001b[0m \u001b[31m4.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading gitdb-4.0.11-py3-none-any.whl (62 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m3.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading smmap-5.0.1-py3-none-any.whl (24 kB)\n",
            "Installing collected packages: watchdog, tenacity, smmap, pydeck, gitdb, gitpython, streamlit\n",
            "  Attempting uninstall: tenacity\n",
            "    Found existing installation: tenacity 9.0.0\n",
            "    Uninstalling tenacity-9.0.0:\n",
            "      Successfully uninstalled tenacity-9.0.0\n",
            "Successfully installed gitdb-4.0.11 gitpython-3.1.43 pydeck-0.9.1 smmap-5.0.1 streamlit-1.38.0 tenacity-8.5.0 watchdog-4.0.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import streamlit as st\n",
        "\n",
        "# Load the data\n",
        "file_path = '/content/amazon.csv'\n",
        "df = pd.read_csv(file_path)\n",
        "\n",
        "# Display the dataframe\n",
        "st.title(\"Amazon Sales Dashboard\")\n",
        "st.write(df.head())\n",
        "\n",
        "# Display basic data overview\n",
        "st.subheader(\"Data Overview\")\n",
        "st.write(\"Total Rows:\", df.shape[0])\n",
        "st.write(\"Total Columns:\", df.shape[1])\n",
        "st.write(df.columns)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-O--IKIQ4kou",
        "outputId": "cc3154e4-88ad-48a0-af26-e9f93dddbac9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-28 16:54:50.517 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.519 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.526 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.527 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.531 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.533 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.537 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.540 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.542 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.545 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.546 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.548 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.549 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.550 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.553 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:54:50.554 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.head)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uTmf7dX_6WJ8",
        "outputId": "b20b2d78-b1a7-450b-cd5c-0409954a001a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<bound method NDFrame.head of       product_id                                       product_name  \\\n",
            "0     B07JW9H4J1  Wayona Nylon Braided USB to Lightning Fast Cha...   \n",
            "1     B098NS6PVG  Ambrane Unbreakable 60W / 3A Fast Charging 1.5...   \n",
            "2     B096MSW6CT  Sounce Fast Phone Charging Cable & Data Sync U...   \n",
            "3     B08HDJ86NZ  boAt Deuce USB 300 2 in 1 Type-C & Micro USB S...   \n",
            "4     B08CF3B7N1  Portronics Konnect L 1.2M Fast Charging 3A 8 P...   \n",
            "...          ...                                                ...   \n",
            "1460  B08L7J3T31  Noir Aqua - 5pcs PP Spun Filter + 1 Spanner | ...   \n",
            "1461  B01M6453MB  Prestige Delight PRWO Electric Rice Cooker (1 ...   \n",
            "1462  B009P2LIL4  Bajaj Majesty RX10 2000 Watts Heat Convector R...   \n",
            "1463  B00J5DYCCA  Havells Ventil Air DSP 230mm Exhaust Fan (Pist...   \n",
            "1464  B01486F4G6  Borosil Jumbo 1000-Watt Grill Sandwich Maker (...   \n",
            "\n",
            "                                               category discounted_price  \\\n",
            "0     Computers&Accessories|Accessories&Peripherals|...             ₹399   \n",
            "1     Computers&Accessories|Accessories&Peripherals|...             ₹199   \n",
            "2     Computers&Accessories|Accessories&Peripherals|...             ₹199   \n",
            "3     Computers&Accessories|Accessories&Peripherals|...             ₹329   \n",
            "4     Computers&Accessories|Accessories&Peripherals|...             ₹154   \n",
            "...                                                 ...              ...   \n",
            "1460  Home&Kitchen|Kitchen&HomeAppliances|WaterPurif...             ₹379   \n",
            "1461  Home&Kitchen|Kitchen&HomeAppliances|SmallKitch...           ₹2,280   \n",
            "1462  Home&Kitchen|Heating,Cooling&AirQuality|RoomHe...           ₹2,219   \n",
            "1463  Home&Kitchen|Heating,Cooling&AirQuality|Fans|E...           ₹1,399   \n",
            "1464  Home&Kitchen|Kitchen&HomeAppliances|SmallKitch...           ₹2,863   \n",
            "\n",
            "     actual_price discount_percentage rating rating_count  \\\n",
            "0          ₹1,099                 64%    4.2       24,269   \n",
            "1            ₹349                 43%    4.0       43,994   \n",
            "2          ₹1,899                 90%    3.9        7,928   \n",
            "3            ₹699                 53%    4.2       94,363   \n",
            "4            ₹399                 61%    4.2       16,905   \n",
            "...           ...                 ...    ...          ...   \n",
            "1460         ₹919                 59%      4        1,090   \n",
            "1461       ₹3,045                 25%    4.1        4,118   \n",
            "1462       ₹3,080                 28%    3.6          468   \n",
            "1463       ₹1,890                 26%      4        8,031   \n",
            "1464       ₹3,690                 22%    4.3        6,987   \n",
            "\n",
            "                                          about_product  \\\n",
            "0     High Compatibility : Compatible With iPhone 12...   \n",
            "1     Compatible with all Type C enabled devices, be...   \n",
            "2     【 Fast Charger& Data Sync】-With built-in safet...   \n",
            "3     The boAt Deuce USB 300 2 in 1 cable is compati...   \n",
            "4     [CHARGE & SYNC FUNCTION]- This cable comes wit...   \n",
            "...                                                 ...   \n",
            "1460  SUPREME QUALITY 90 GRAM 3 LAYER THIK PP SPUN F...   \n",
            "1461                       230 Volts, 400 watts, 1 Year   \n",
            "1462  International design and styling|Two heat sett...   \n",
            "1463  Fan sweep area: 230 MM ; Noise level: (40 - 45...   \n",
            "1464  Brand-Borosil, Specification â€“ 23V ~ 5Hz;1 W...   \n",
            "\n",
            "                                                user_id  \\\n",
            "0     AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBB...   \n",
            "1     AECPFYFQVRUWC3KGNLJIOREFP5LQ,AGYYVPDD7YG7FYNBX...   \n",
            "2     AGU3BBQ2V2DDAMOAKGFAWDDQ6QHA,AESFLDV2PT363T2AQ...   \n",
            "3     AEWAZDZZJLQUYVOVGBEUKSLXHQ5A,AG5HTSFRRE6NL3M5S...   \n",
            "4     AE3Q6KSUK5P75D5HFYHCRAOLODSA,AFUGIFH5ZAFXRDSZH...   \n",
            "...                                                 ...   \n",
            "1460  AHITFY6AHALOFOHOZEOC6XBP4FEA,AFRABBODZJZQB6Z4U...   \n",
            "1461  AFG5FM3NEMOL6BNFRV2NK5FNJCHQ,AGEINTRN6Z563RMLH...   \n",
            "1462  AGVPWCMAHYQWJOQKMUJN4DW3KM5Q,AF4Q3E66MY4SR7YQZ...   \n",
            "1463  AF2JQCLSCY3QJATWUNNHUSVUPNQQ,AFDMLUXC5LS5RXDJS...   \n",
            "1464  AFGW5PT3R6ZAVQR4Y5MWVAKBZAYA,AG7QNJ2SCS5VS5VYY...   \n",
            "\n",
            "                                              user_name  \\\n",
            "0     Manav,Adarsh gupta,Sundeep,S.Sayeed Ahmed,jasp...   \n",
            "1     ArdKn,Nirbhay kumar,Sagar Viswanathan,Asp,Plac...   \n",
            "2     Kunal,Himanshu,viswanath,sai niharka,saqib mal...   \n",
            "3     Omkar dhale,JD,HEMALATHA,Ajwadh a.,amar singh ...   \n",
            "4     rahuls6099,Swasat Borah,Ajay Wadke,Pranali,RVK...   \n",
            "...                                                 ...   \n",
            "1460  Prabha ds,Raghuram bk,Real Deal,Amazon Custome...   \n",
            "1461  Manu Bhai,Naveenpittu,Evatira Sangma,JAGANNADH...   \n",
            "1462  Nehal Desai,Danish Parwez,Amazon Customer,Amaz...   \n",
            "1463  Shubham Dubey,E.GURUBARAN,Mayank S.,eusuf khan...   \n",
            "1464  Rajib,Ajay B,Vikas Kahol,PARDEEP,Anindya Prama...   \n",
            "\n",
            "                                              review_id  \\\n",
            "0     R3HXWT0LRP0NMF,R2AJM3LFTLZHFO,R6AQJGUP6P86,R1K...   \n",
            "1     RGIQEG07R9HS2,R1SMWZQ86XIN8U,R2J3Y1WL29GWDE,RY...   \n",
            "2     R3J3EQQ9TZI5ZJ,R3E7WBGK7ID0KV,RWU79XKQ6I1QF,R2...   \n",
            "3     R3EEUZKKK9J36I,R3HJVYCLYOY554,REDECAZ7AMPQC,R1...   \n",
            "4     R1BP4L2HH9TFUP,R16PVJEXKV6QZS,R2UPDB81N66T4P,R...   \n",
            "...                                                 ...   \n",
            "1460  R3G3XFHPBFF0E8,R3C0BZCD32EIGW,R2EBVBCN9QPD9R,R...   \n",
            "1461  R3DDL2UPKQ2CK9,R2SYYU1OATVIU5,R1VM993161IYRW,R...   \n",
            "1462  R1TLRJVW4STY5I,R2O455KRN493R1,R3Q5MVGBRIAS2G,R...   \n",
            "1463  R39Q2Y79MM9SWK,R3079BG1NIH6MB,R29A31ZELTZNJM,R...   \n",
            "1464  R20RBRZ0WEUJT9,ROKIFK9R2ISSE,R30EEG2FNJSN5I,R2...   \n",
            "\n",
            "                                           review_title  \\\n",
            "0     Satisfied,Charging is really fast,Value for mo...   \n",
            "1     A Good Braided Cable for Your Type C Device,Go...   \n",
            "2     Good speed for earlier versions,Good Product,W...   \n",
            "3     Good product,Good one,Nice,Really nice product...   \n",
            "4     As good as original,Decent,Good one for second...   \n",
            "...                                                 ...   \n",
            "1460  Received the product without spanner,Excellent...   \n",
            "1461  ok,everything was good couldn't return bcoz I ...   \n",
            "1462  very good,Work but front melt after 2 month,Go...   \n",
            "1463  Fan Speed is slow,Good quality,Good product,go...   \n",
            "1464  Works perfect,Ok good product,Nice Product. Re...   \n",
            "\n",
            "                                         review_content  \\\n",
            "0     Looks durable Charging is fine tooNo complains...   \n",
            "1     I ordered this cable to connect my phone to An...   \n",
            "2     Not quite durable and sturdy,https://m.media-a...   \n",
            "3     Good product,long wire,Charges good,Nice,I bou...   \n",
            "4     Bought this instead of original apple, does th...   \n",
            "...                                                 ...   \n",
            "1460  I received product without spanner,Excellent p...   \n",
            "1461  ok,got everything as mentioned but the measuri...   \n",
            "1462  plastic but cool body ,u have to find sturdy s...   \n",
            "1463  I have installed this in my kitchen working fi...   \n",
            "1464  It does it job perfectly..only issue is temp c...   \n",
            "\n",
            "                                               img_link  \\\n",
            "0     https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "1     https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "2     https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "3     https://m.media-amazon.com/images/I/41V5FtEWPk...   \n",
            "4     https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "...                                                 ...   \n",
            "1460  https://m.media-amazon.com/images/I/41fDdRtjfx...   \n",
            "1461  https://m.media-amazon.com/images/I/41gzDxk4+k...   \n",
            "1462  https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "1463  https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "1464  https://m.media-amazon.com/images/W/WEBP_40237...   \n",
            "\n",
            "                                           product_link  \n",
            "0     https://www.amazon.in/Wayona-Braided-WN3LG1-Sy...  \n",
            "1     https://www.amazon.in/Ambrane-Unbreakable-Char...  \n",
            "2     https://www.amazon.in/Sounce-iPhone-Charging-C...  \n",
            "3     https://www.amazon.in/Deuce-300-Resistant-Tang...  \n",
            "4     https://www.amazon.in/Portronics-Konnect-POR-1...  \n",
            "...                                                 ...  \n",
            "1460  https://www.amazon.in/Noir-Aqua-Spanner-Purifi...  \n",
            "1461  https://www.amazon.in/Prestige-Delight-PRWO-1-...  \n",
            "1462  https://www.amazon.in/Bajaj-RX-10-2000-Watt-Co...  \n",
            "1463  https://www.amazon.in/Havells-Ventilair-230mm-...  \n",
            "1464  https://www.amazon.in/Borosil-Jumbo-1000-Watt-...  \n",
            "\n",
            "[1465 rows x 16 columns]>\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        " 3: Data Cleaning and Preprocessing"
      ],
      "metadata": {
        "id": "RFRrwbJf5MBD"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Convert columns to numeric where applicable\n",
        "df['discounted_price'] = pd.to_numeric(df['discounted_price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')\n",
        "df['actual_price'] = pd.to_numeric(df['actual_price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')\n",
        "df['discount_percentage'] = pd.to_numeric(df['discount_percentage'].str.replace('%', ''), errors='coerce')\n",
        "df['rating'] = pd.to_numeric(df['rating'], errors='coerce')\n",
        "df['rating_count'] = pd.to_numeric(df['rating_count'].str.replace(',', ''), errors='coerce')\n",
        "\n",
        "# Handle missing values\n",
        "df.dropna(inplace=True)\n",
        "\n",
        "st.write(\"Cleaned Data\")\n",
        "st.write(df.head())\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zDWtD-tD5M7M",
        "outputId": "fed6ece7-6497-474f-9220-4852ac243cfb"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-28 16:56:14.602 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:14.603 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:14.606 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:14.609 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:14.616 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:14.618 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "3: Data Analysis and Visualization"
      ],
      "metadata": {
        "id": "t3tqhJPN6t53"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Price Distribution\n",
        "st.subheader('Price Distribution')\n",
        "fig, ax = plt.subplots()\n",
        "sns.histplot(df['discounted_price'], bins=30, ax=ax)\n",
        "ax.set_title('Distribution of Discounted Prices')\n",
        "st.pyplot(fig)\n",
        "\n",
        "# Discount Percentage vs Rating\n",
        "st.subheader('Discount vs Rating')\n",
        "fig, ax = plt.subplots()\n",
        "sns.scatterplot(data=df, x='discount_percentage', y='rating', ax=ax)\n",
        "ax.set_title('Discount Percentage vs Rating')\n",
        "st.pyplot(fig)\n",
        "\n",
        "# Average Rating by Category\n",
        "st.subheader('Average Rating by Category')\n",
        "avg_rating_by_category = df.groupby('category')['rating'].mean().sort_values(ascending=False)\n",
        "st.bar_chart(avg_rating_by_category)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "m7DvyArL6uzN",
        "outputId": "5ead0a82-4aaf-46c6-f1a9-cefc56998956"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-28 16:56:49.528 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:49.534 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:49.867 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:50.609 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:50.616 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:50.632 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:50.635 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:50.893 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:51.484 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:51.496 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:51.498 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:51.503 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:52.360 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:56:52.362 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "DeltaGenerator()"
            ]
          },
          "metadata": {},
          "execution_count": 17
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAk8AAAHHCAYAAACmzLxGAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABKkElEQVR4nO3deVwVdf///+dB4IAiICqbIZKaWy7lQqRtSpKabXaVXWRapi2YWx8zK5css7TSNNO6rlJbbE8rMxP3FiOXUFEkLVwuE9AQEBdAeP/+6Mv8PILmIAri4367ze3mmfd7Zl7vmSM8mTMzx2GMMQIAAMAZcavoAgAAAC4khCcAAAAbCE8AAAA2EJ4AAABsIDwBAADYQHgCAACwgfAEAABgA+EJAADABsITAACADYQn4DTGjRsnh8NxXrZ1/fXX6/rrr7der1y5Ug6HQ5999tl52X6/fv3UoEGD87KtssrNzdWDDz6o4OBgORwODR06tFzX73A4NG7cuHJdZ1V28nv2XNm5c6ccDofmzJlzzrcFnAnCEy4ac+bMkcPhsCYvLy+FhoYqJiZG06ZN06FDh8plO3/++afGjRunxMTEcllfearMtZ2JF154QXPmzNEjjzyi9957T3369Dll3wYNGljH2s3NTf7+/mrZsqUGDhyohISE81h1xfrpp580btw4ZWVlVVgNJx4Lh8OhwMBAXXPNNZo/f36F1QScDQffbYeLxZw5c3T//fdr/PjxioiIUEFBgdLS0rRy5UrFx8erfv36+uqrr9SqVStrmePHj+v48ePy8vI64+2sW7dO7du31+zZs9WvX78zXi4/P1+S5OnpKenvM0833HCDPv30U915551nvJ6y1lZQUKCioiI5nc5y2da5cNVVV8nd3V0//PDDP/Zt0KCBatWqpccff1ySdOjQISUnJ+vTTz9VWlqahg0bpldffdVlmWPHjsnd3V3u7u7npP6K8PLLL2vEiBFKTU0t9zOLxWedVq5cedp+Jx+LP//8U2+++ab++OMPzZw5Uw8//PBplzfGKC8vTx4eHqpWrVp5lA6clarzEwI4Q926dVO7du2s16NGjdLy5ct1880365ZbblFycrK8vb0l6bz8Ij1y5IiqV69uhaaK4uHhUaHbPxMZGRlq3rz5GfevV6+e7r33Xpd5L730kv79739rypQpaty4sR555BGrzU5Ihj0nH4v77rtPjRo10pQpU04Zno4fP66ioiJ5enpybFCp8LEdIKlz584aPXq0du3apffff9+aX9o1T/Hx8erUqZP8/f3l4+OjJk2a6KmnnpL091/g7du3lyTdf//91scUxddqXH/99br88su1fv16XXvttapevbq17KmuHyksLNRTTz2l4OBg1ahRQ7fccov27Nnj0qdBgwalnuU6cZ3/VFtp1zwdPnxYjz/+uMLCwuR0OtWkSRO9/PLLOvmEtcPh0KBBg7RgwQJdfvnlcjqdatGihRYvXlz6Dj9JRkaG+vfvr6CgIHl5eal169aaO3eu1V58/Vdqaqq++eYbq/adO3ee0fpP5O3trffee08BAQGaMGGCy1hOvubp0KFDGjp0qBo0aCCn06nAwEDdeOON2rBhg8s6ExIS1L17d9WqVUs1atRQq1at9Nprr7n0Wb58ua655hrVqFFD/v7+uvXWW5WcnOzS51TXnZX2PjyTfT5u3DiNGDFCkhQREVHqfnv//ffVtm1beXt7KyAgQL179y7x/pKkt956Sw0bNpS3t7c6dOig77//vvQdfIaCg4PVrFkzpaamSvr/r2t6+eWXNXXqVDVs2FBOp1Nbt2495TVP27Zt01133aW6devK29tbTZo00dNPP+3SZ+/evXrggQcUFBRk7aN33nmnRD3Tp09XixYtVL16ddWqVUvt2rXTvHnzzmqMqLo48wT8P3369NFTTz2lJUuWaMCAAaX22bJli26++Wa1atVK48ePl9Pp1I4dO/Tjjz9Kkpo1a6bx48drzJgxGjhwoK655hpJ0tVXX22t46+//lK3bt3Uu3dv3XvvvQoKCjptXRMmTJDD4dDIkSOVkZGhqVOnKjo6WomJidYZsjNxJrWdyBijW265RStWrFD//v3Vpk0bfffddxoxYoT27t2rKVOmuPT/4Ycf9MUXX+jRRx9VzZo1NW3aNPXq1Uu7d+9W7dq1T1nX0aNHdf3112vHjh0aNGiQIiIi9Omnn6pfv37KysrSkCFD1KxZM7333nsaNmyYLrnkEuvjn7p1657x+E/k4+Oj22+/XW+//ba2bt2qFi1alNrv4Ycf1meffaZBgwapefPm+uuvv/TDDz8oOTlZV155paS/w/TNN9+skJAQDRkyRMHBwUpOTtbChQs1ZMgQSdLSpUvVrVs3XXrppRo3bpyOHj2q6dOnq2PHjtqwYUOZP077p31+xx136LffftOHH36oKVOmqE6dOpL+//02YcIEjR49WnfddZcefPBB7d+/X9OnT9e1116rX3/9Vf7+/pKkt99+Ww899JCuvvpqDR06VH/88YduueUWBQQEKCwsrEy1FxQUaM+ePSXeG7Nnz9axY8c0cOBAOZ1OBQQEqKioqMTymzZt0jXXXCMPDw8NHDhQDRo00O+//66vv/5aEyZMkCSlp6frqquusoJm3bp19e2336p///7Kycmxbjj4z3/+o8GDB+vOO+/UkCFDdOzYMW3atEkJCQn697//XabxoYozwEVi9uzZRpJZu3btKfv4+fmZK664wno9duxYc+J/kylTphhJZv/+/adcx9q1a40kM3v27BJt1113nZFkZs2aVWrbddddZ71esWKFkWTq1atncnJyrPmffPKJkWRee+01a154eLjp27fvP67zdLX17dvXhIeHW68XLFhgJJnnn3/epd+dd95pHA6H2bFjhzVPkvH09HSZt3HjRiPJTJ8+vcS2TjR16lQjybz//vvWvPz8fBMVFWV8fHxcxh4eHm569Ohx2vWdad/iY/nll1+6jGPs2LHWaz8/PxMXF3fKdRw/ftxERESY8PBwc/DgQZe2oqIi699t2rQxgYGB5q+//rLmbdy40bi5uZn77rvPmnfyMSh28vuwuNYz2eeTJ082kkxqaqrL8jt37jTVqlUzEyZMcJm/efNm4+7ubs3Pz883gYGBpk2bNiYvL8/q99ZbbxlJLu+vUwkPDzddu3Y1+/fvN/v37zcbN240vXv3NpLMY489ZowxJjU11Ugyvr6+JiMjw2X54rYT37fXXnutqVmzptm1a5dL3xP3e//+/U1ISIg5cOCAS5/evXsbPz8/c+TIEWOMMbfeeqtp0aLFP44DKMbHdsAJfHx8TnvXXfFf4l9++WWpfw2fCafTqfvvv/+M+993332qWbOm9frOO+9USEiIFi1aVKbtn6lFixapWrVqGjx4sMv8xx9/XMYYffvtty7zo6Oj1bBhQ+t1q1at5Ovrqz/++OMftxMcHKx77rnHmufh4aHBgwcrNzdXq1atKofRlOTj4yNJ/3i8ExIS9Oeff5ba/uuvvyo1NVVDhw613hvFij9m27dvnxITE9WvXz8FBARY7a1atdKNN954VsexrPtckr744gsVFRXprrvu0oEDB6wpODhYjRs31ooVKyT9fZNBRkaGHn74YZfr8vr16yc/P78zrnXJkiWqW7eu6tatq9atW+vTTz9Vnz599NJLL7n069Wr1z+eUdy/f79Wr16tBx54QPXr13dpK97vxhh9/vnn6tmzp4wxLmOMiYlRdna29fGrv7+//ve//2nt2rVnPB5c3AhPwAlyc3NdgsrJ7r77bnXs2FEPPviggoKC1Lt3b33yySe2glS9evVsXRzeuHFjl9cOh0ONGjUq0/U+duzatUuhoaEl9kezZs2s9hOd/EtMkmrVqqWDBw/+43YaN24sNzfXH0en2k55yc3NlaTTHu9JkyYpKSlJYWFh6tChg8aNG+cSTH7//XdJ0uWXX37KdRTX36RJkxJtzZo104EDB3T48OEyjaGs+1yStm/fLmOMGjdubIWa4ik5OVkZGRku9Z/8PvTw8NCll156xrVGRkYqPj5eS5cu1U8//aQDBw7o3XffLfHRc0RExD+uq/gYnG6/79+/X1lZWXrrrbdKjK/4j5fiMY4cOVI+Pj7q0KGDGjdurLi4OOujeKA0XPME/D//+9//lJ2drUaNGp2yj7e3t1avXq0VK1bom2++0eLFi/Xxxx+rc+fOWrJkyRndRm3nOqUzdaoHeRYWFp63W7tPtR1TSZ+GkpSUJEmnPd533XWX9TyiJUuWaPLkyXrppZf0xRdfqFu3buVe0+mOY2nOZp8XFRXJ4XDo22+/LXU9xWfmykudOnUUHR39j/3K6/9H8R809957r/r27Vtqn+LHkjRr1kwpKSlauHChFi9erM8//1xvvPGGxowZo2effbZc6kHVQngC/p/33ntPkhQTE3Pafm5uburSpYu6dOmiV199VS+88IKefvpprVixQtHR0eX+RPLt27e7vDbGaMeOHS7Po6pVq1apD0HctWuXy9kBO7WFh4dr6dKlOnTokMvZmW3btlnt5SE8PFybNm1SUVGRy9mn8t7OiXJzczV//nyFhYVZZ7hOJSQkRI8++qgeffRRZWRk6Morr9SECRPUrVs36yOzpKSkUwaD4vpTUlJKtG3btk116tRRjRo1JJ3+OJbVqY55w4YNZYxRRESELrvsslMuX1z/9u3b1blzZ2t+QUGBUlNT1bp16zLXVlbF7+niAFyaunXrqmbNmiosLDyj0FajRg3dfffduvvuu5Wfn6877rhDEyZM0KhRo3hMAkrgYztAf99G/txzzykiIkKxsbGn7JeZmVliXps2bSRJeXl5kmT9IiyvJzq/++67LtflfPbZZ9q3b5/LmY+GDRvq559/th60KUkLFy4sccu5ndq6d++uwsJCvf766y7zp0yZIofDUW5nXrp37660tDR9/PHH1rzjx49r+vTp8vHx0XXXXVcu2yl29OhR9enTR5mZmXr66adPe7YnOzvbZV5gYKBCQ0OtY33llVcqIiJCU6dOLbFPi8/+hISEqE2bNpo7d65Ln6SkJC1ZskTdu3e35jVs2FDZ2dnatGmTNW/fvn1n9STuUx3zO+64Q9WqVdOzzz5b4kyVMUZ//fWXJKldu3aqW7euZs2a5fL+mjNnToU9tbxu3bq69tpr9c4772j37t0ubcVjqVatmnr16qXPP/+81JC1f/9+69/FYy3m6emp5s2byxijgoKCczACXOg484SLzrfffqtt27bp+PHjSk9P1/LlyxUfH6/w8HB99dVXp/0rc/z48Vq9erV69Oih8PBwZWRk6I033tAll1yiTp06Sfr7F6C/v79mzZqlmjVrqkaNGoqMjDyjazlKExAQoE6dOun+++9Xenq6pk6dqkaNGrk8TuHBBx/UZ599pptuukl33XWXfv/9d73//vsuFxPbra1nz5664YYb9PTTT2vnzp1q3bq1lixZoi+//FJDhw4tse6yGjhwoN58803169dP69evV4MGDfTZZ5/pxx9/1NSpU097TdI/2bt3r/XcrtzcXG3dutV6wvjjjz+uhx566JTLHjp0SJdcconuvPNOtW7dWj4+Plq6dKnWrl2rV155RdLfZyFnzpypnj17qk2bNrr//vsVEhKibdu2acuWLfruu+8kSZMnT1a3bt0UFRWl/v37W48q8PPzc3muVO/evTVy5EjdfvvtGjx4sI4cOaKZM2fqsssuK/FsqTPVtm1bSdLTTz+t3r17y8PDQz179lTDhg31/PPPa9SoUdq5c6duu+021axZU6mpqZo/f74GDhyo//u//5OHh4eef/55PfTQQ+rcubPuvvtupaamavbs2baueSpv06ZNU6dOnXTllVdq4MCBioiI0M6dO/XNN99YXz/04osvasWKFYqMjNSAAQPUvHlzZWZmasOGDVq6dKn1x1DXrl0VHBysjh07KigoSMnJyXr99dfVo0ePs3r/oQqrkHv8gApQ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pYaUlBTt3r3bZaybN292+U8dHx8vX19f64f/P43jfL8nioqKlJeXV6XG16VLF23evFmJiYnW1K5dO8XGxlr/ripjPVFubq5+//13hYSEVKnj2bFjxxKPDfntt98UHh4uqWr9HJKk2bNnKzAwUD169LDmVaXjeeTIEbm5uUaOatWqqaioSNIFeDzP+NJylKuPPvrIOJ1OM2fOHLN161YzcOBA4+/v73LHREU6dOiQ+fXXX82vv/5qJJlXX33V/Prrr2bXrl3GmL9vKfX39zdffvml2bRpk7n11ltLvaX0iiuuMAkJCeaHH34wjRs3drmlNCsrywQFBZk+ffqYpKQk89FHH5nq1auXuKXU3d3dvPzyyyY5OdmMHTu2XB9V8Mgjjxg/Pz+zcuVKl9uFjxw5YvV5+OGHTf369c3y5cvNunXrTFRUlImKirLai28V7tq1q0lMTDSLFy82devWLfVW4REjRpjk5GQzY8aMUm8VPhfviSeffNKsWrXKpKammk2bNpknn3zSOBwOs2TJkioxvtM58W67qjLWxx9/3KxcudKkpqaaH3/80URHR5s6deqYjIyMKjNGY/5+3IS7u7uZMGGC2b59u/nggw9M9erVzfvvv2/1qSo/hwoLC039+vXNyJEjS7RVlePZt29fU69ePetRBV988YWpU6eOeeKJJ6w+F9LxJDxVoOnTp5v69esbT09P06FDB/Pzzz9XdEmWFStWGEklpr59+xpj/r6tdPTo0SYoKMg4nU7TpUsXk5KS4rKOv/76y9xzzz3Gx8fH+Pr6mvvvv98cOnTIpc/GjRtNp06djNPpNPXq1TMvvvhiiVo++eQTc9lllxlPT0/TokUL880335TbOEsboyQze/Zsq8/Ro0fNo48+amrVqmWqV69ubr/9drNv3z6X9ezcudN069bNeHt7mzp16pjHH3/cFBQUuPRZsWKFadOmjfH09DSXXnqpyzaKnYv3xAMPPGDCw8ONp6enqVu3runSpYsVnKrC+E7n5PBUFcZ69913m5CQEOPp6Wnq1atn7r77bpdnH1WFMRb7+uuvzeWXX26cTqdp2rSpeeutt1zaq8rPoe+++85IKlG7MVXneObk5JghQ4aY+vXrGy8vL3PppZeap59+2uWRAhfS8XQYc8LjPQEAAHBaXPMEAABgA+EJAADABsITAACADYQnAAAAGwhPAAAA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\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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f/7zH5x//vm46KKLMGvWLEyePBkDBgzAtm3b8Prrr+OUU07xObdz587IysrC888/j4yMDKSlpeHss8/2u+cqUGbPno1LLrkEAwcOxOTJk3H8+HFvzkUbbNLICc9DWoSEHvVRcPWVlJSktGzZUvnd736nPP3000pRUZHfOfpHwVetWqVccsklSuvWrZWkpCSldevWyvjx45Xvv//e57yysjLlnnvuUTp16qQkJiYqLVu2VC6//HLlhx9+8JYpLi5W/vznPyutW7dWEhMTlby8PGXevHk+j1KrdV177bVKZmamkpGRoYwbN045fPiw4aPg+sfU1X7v2bPHe+y7775ThgwZoqSkpCgATB8LVx/PlXm0ePPmzcqll16qNG3aVHG73UqHDh2UcePGKatWrbKMU2XhwoVKnz59FLfbrWRnZytDhw5VVq5c6VPm448/VkaOHKlkZmYqycnJSufOnZVJkyYpX331lbfMxIkTlbS0NL/69WOqKIry2WefKf369VOSkpJ88vrzzz8rY8eOVbKyspTMzEzliiuuUA4cOCB8dHzVqlVKnz59lKSkJKVz587Kiy++qNx2221KcnKyXwxLly5VBg0apKSlpSlpaWlK165dlSlTpig7d+40za/KypUrFQBKXFycsn//fp/3jh49qkyZMkXp2rWrkpaWpmRmZipnn3228tZbb1nWazY2hYWFSmZmpjJ06FBFUeoeBb/tttuUVq1aKSkpKcrAgQOV9evXK0OHDvWWUVmxYoXSrVs376Px6mPhRo+Ci+aaKOdLlixRunbtqrjdbqVHjx7Ku+++q1x22WVK165dLftKGhdxisI7tgghxAnGjBmDHTt2+N2nQ0JH79690axZM6xcuTLcoZAIgvfcEEKIDU6cOOHz965du/D+++/TjDNEVFVVobq62ufYmjVr8PXXXzPnxA9+c0MIITZo1aqV14fqp59+wvz581FRUYHNmzcb6toQ++zduxcjRozAH/7wB7Ru3Rrfffcdnn/+eWRmZmL79u1Ba+2Q2II3FBNCiA3OP/98vPHGGzh48CDcbjf69++P2bNnc2MTIrKzs9GvXz+8+OKLOHLkCNLS0jBq1CjMmTOHGxviB7+5IYQQQkhMwXtuCCGEEBJTcHNDCCGEkJii0d1zU1tbiwMHDiAjIyMkMuGEEEIIcR5FUVBcXIzWrVt7Pc+MaHSb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jh0BAGPHjkVcXJz37x9++AGXXHIJWrRogfT0dJx55pn46KOPfPr966+/YtSoUUhJSUGnTp2wePFivzEoKCjAddddh2bNmsHj8eC8887D119/bZlTQmIFbm4IiULuuOMOrF27FitWrMCHH36INWvWYNOmTcKyS5cuxZNPPokFCxZg165dWL58OXr27Ol9f8KECXjjjTfwzDPP4Ntvv8WCBQuQnp4OAPjll19w4YUX4swzz8TXX3+N+fPn46WXXsLDDz8ccMwzZ87EuHHjsHXrVlx44YW4+uqr8dtvv6Fdu3ZYunQpAGDnzp349ddf8fTTT0vV+corr8DlcuHLL7/E008/jSeeeAIvvvii9/2pU6di/fr1WLJkCbZu3YorrrgC559/Pnbt2uUtU1ZWhrlz5+LFF1/Ejh070Lx5c9OcFBQU4LzzzkOfPn3w1Vdf4T//+Q8OHTqEcePG+cWWlpaGL774Ao8++ihmzZqFlStXAqj72RAAXn75Zfz666/ev0tKSnDhhRdi1apV2Lx5M84//3yMHj0a+/bt89Y7YcIEHDhwAGvWrMHSpUvxwgsv4PDhwz5tX3HFFTh8+DA++OADbNy4EX379sXw4cPx22+/SeWVkKgnvL6dhJBAKS4uVpKSkpS33nrLe+zYsWNKSkqK1527Q4cOypNPPqkoiqI8/vjjyqmnnqpUVlb61bVz504FgLJy5UphW3fffbfSpUsXpba21nvsb3/7m5Kenq7U1NT4taVy+umnKw888ID3bwDKvffe6/27pKREAaB88MEHiqIoyscff6wAUI4fPy6bBmXo0KHKaaed5hPbnXfeqZx22mmKoijKTz/9pCQkJCi//PKLz3nDhw9XZsyYoSiKorz88ssKAGXLli3e961y8tBDDym///3vfY7t379fAeB1fx46dKgyaNAgnzJnnnmmcuedd3r/BqAsW7bMsp/du3dXnn32WUVRFOXbb79VACgbNmzwvr9r1y4FgHcM/ve//ykej0cpLy/3qadz587KggULLNsjJBbgNzeERBk//PADKisrcfbZZ3uPNWnSBF26dBGWv+KKK3DixAmccsopuP7667Fs2TLvz0pbtmxBQkIChg4dKjz322+/Rf/+/REXF+c9NnDgQJSUlODnn38OKO5evXp5/z8tLQ0ej8fvG4dAOeecc3xi69+/P3bt2oWamhps27YNNTU1OPXUU5Genu59rV27Fj/88IP3nKSkJJ/YrHLy9ddf4+OPP/aps2vXrgDgU6+2TgBo1aqVZX9LSkpw++2347TTTkNWVhbS09Px7bffer+52blzJ1wuF/r27es9Jzc3F9nZ2T7xlZSUoGnTpj4x7tmzxyc+QmIZ3lBMSIzTrl077Ny5Ex999BFWrlyJP/3pT5g3bx7Wrl2LlJSUoOuPj4/3uc8FAKqqqvzKJSYm+vwdFxeH2traoNs3oqSkBAkJCdi4cSMSEhJ83lN/YgKAlJQUnw2SVU5KSkowevRozJ071++9Vq1aef/fTn9vv/12rFy5Eo899hhyc3ORkpKCyy+/HJWVlabn6eNr1aqVz71FKllZWdL1EBLNcHNDSJTRuXNnJCYm4osvvkD79u0BAMePH8f3339v+G1DSkoKRo8ejdGjR2PKlCno2rUrtm3bhp49e6K2thZr167FiBEj/M477bTTsHTpUiiK4t0AfPrpp8jIyEDbtm0B1D2Z9euvv3rPKSoqwp49ewLqU1JSEoC6p5UC4YsvvvD5+/PPP0deXh4SEhLQp08f1NTU4PDhwxg8eLB0nVY56du3L5YuXYqOHTvC5bK/hCYmJvr199NPP8WkSZMwduxYAHUblb1793rf79KlC6qrq7F582b069cPALB7924cP37cJ76DBw/C5XJ5b1QmpLHBn6UIiTLS09Nx7bXX4o477sDq1auxfft2TJo0yfDx5UWLFuGll17C9u3b8eOPP+K1115DSkoKOnTogI4dO2LixIm45pprsHz5cuzZswdr1qzBW2+9BQD405/+hP379+Pmm2/Gd999hxUrVuCBBx7Arbfe6m3vvPPOwz/+8Q/873//w7Zt2zBx4kS/b0qs6NChA+Li4vDee+/hyJEjKCkpkTpv3759uPXWW7Fz50688cYbePbZZzFt2jQAwKmnnoqrr74aEyZMwDvvvIM9e/bgyy+/RH5+Pv79738b1mmVkylTpuC3337D+PHjsWHDBvzwww/473//i8mTJwe0OevYsSNWrVqFgwcPejcneXl5eOedd7BlyxZ8/fXX+L//+z+fb3u6du2KESNG4IYbbsCXX36JzZs344YbbvD59mnEiBHo378/xowZgw8//BB79+7FZ599hnvuuQdfffWVdHyERDPc3BAShcybNw+DBw/G6NGjMWLECAwaNMj7L3k9WVlZ+Pvf/46BAweiV69e+Oijj/Cvf/0LTZs2BQDMnz8fl19+Of70pz+ha9euuP7661FaWgoAaNOmDd5//318+eWXOP3003HjjTfi2muvxb333uutf8aMGRg6dCguuugijBo1CmPGjEHnzp0D6k+bNm0wc+ZM3HXXXWjRogWmTp0qdd6ECRNw4sQJnHXWWZgyZQqmTZvmI1748ssvY8KECbjtttvQpUsXjBkzBhs2bPB+42WEWU5at26NTz/9FDU1Nfj973+Pnj17Yvr06cjKygpIH+fxxx/HypUr0a5dO/Tp0wcA8MQTTyA7OxsDBgzA6NGjMXLkSJ/7awDg1VdfRYsWLTBkyBCMHTsW119/PTIyMpCcnAyg7uev999/H0OGDMHkyZNx6qmn4qqrrsJPP/2EFi1aSMdHSDQTp+h/LCeEkChg2LBh6N27d6O3mfj555/Rrl07fPTRRxg+fHi4wyEkIuA9N4QQEkWsXr0aJSUl6NmzJ3799Vf85S9/QceOHTFkyJBwh0ZIxMDNDSEk4ti3bx+6detm+P4333zTgNFEFlVVVbj77rvx448/IiMjAwMGDMDrr7/u93QWIY0Z/ixFCIk4qqurfZ4S0hPsk0qEkNiGmxtCCCGExBR8WooQQgghMQU3N4QQQgiJKbi5IYQQQkhMwc0NIYQQQmIKbm4IIYQQElNwc0MIIYSQmIKbG0IIIYTEFNzcEEIIISSm+P/hgfeoZBkifwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "4: Enhance Interactivity"
      ],
      "metadata": {
        "id": "8L29I7Ve63hR"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Filter by Category\n",
        "st.subheader('Filter by Category')\n",
        "category = st.selectbox('Select Category', df['category'].unique())\n",
        "filtered_df = df[df['category'] == category]\n",
        "\n",
        "st.write(f\"Showing data for category: {category}\")\n",
        "st.write(filtered_df[['product_name', 'discounted_price', 'actual_price', 'discount_percentage', 'rating']])\n",
        "\n",
        "# Average Rating for Selected Category\n",
        "st.subheader(f'Average Rating for {category}')\n",
        "avg_rating = filtered_df['rating'].mean()\n",
        "st.write(f'Average Rating: {avg_rating:.2f}')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k9vD7S6v64e1",
        "outputId": "443a5da1-862d-4eb2-a244-f6fea6b48665"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2024-08-28 16:57:26.927 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.929 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.934 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.936 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.942 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.947 Session state does not function when running a script without `streamlit run`\n",
            "2024-08-28 16:57:26.951 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.953 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.960 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.963 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.967 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.968 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.986 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.989 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.991 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.993 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.996 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:26.999 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:27.001 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2024-08-28 16:57:27.002 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    }
  ]
}