import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="LoanTap - Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load the trained model and scaler
with open('loantap_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('loantap_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for insights
df = pd.read_csv('logistic_regression.csv')

# Preprocessing Steps
df.loc[(df.home_ownership == 'ANY') | (df.home_ownership == 'NONE'), 'home_ownership'] = 'OTHER'
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
df['issue_d'] = pd.to_datetime(df['issue_d'])
df['Credit_History_Years'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25

# Extract expected feature names from scaler
feature_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else df.drop(columns=['loan_status']).columns

# Define categorical feature options
term_options = [36, 60]
purpose_options = df['purpose'].unique().tolist()
verification_status_options = df['verification_status'].unique().tolist()
grade_options = df['grade'].unique().tolist()
home_ownership_options = df['home_ownership'].unique().tolist()

# ---- Sidebar with Information ----
st.sidebar.image("data:image/svg+xml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA3ODQuOTggMTU2LjExIj48ZGVmcz48c3R5bGU+LmNscy0xe2ZpbGw6Izk5M2Y5MDt9LmNscy0ye2ZpbGw6IzYyNDE5ODt9LmNscy0ze2ZpbGw6IzU2NGM1ODt9PC9zdHlsZT48L2RlZnM+PHBvbHlnb24gY2xhc3M9ImNscy0xIiBwb2ludHM9IjMuMjkgMi4yNyAzLjI5IDUyLjU5IDEwNC4zOSA1Mi41OSAxMDQuMzkgMTUzLjY4IDE1NC43IDE1My42OCAxNTQuNyAyLjI3IDMuMjkgMi4yNyIvPjxwb2x5Z29uIGNsYXNzPSJjbHMtMiIgcG9pbnRzPSIzMi4wMiAxMjQuOTYgMzIuMDIgODguMDggMy4yOSA2Ny40IDMuMjkgMTUzLjY4IDg5LjU4IDE1My42OCA2OC45IDEyNC45NiAzMi4wMiAxMjQuOTYiLz48cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yMDgsODkuODhIMjQyLjN2MTYuNTZIMTg3LjA4VjIuMjdIMjA4WiIvPjxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTI2Ni42NiwxMDIuNDlhMzguMSwzOC4xLDAsMCwxLTE1LTE1LDQ0Ljg3LDQ0Ljg3LDAsMCwxLTUuNDUtMjIuMzksNDQsNDQsMCwwLDEsNS42LTIyLjM4LDM4LjgyLDM4LjgyLDAsMCwxLDE1LjI5LTE1LDQ2LjgzLDQ2LjgzLDAsMCwxLDQzLjI4LDAsMzguODQsMzguODQsMCwwLDEsMTUuMywxNSw0NCw0NCwwLDAsMSw1LjYsMjIuMzgsNDMuMTEsNDMuMTEsMCwwLDEtNS43NSwyMi4zOSwzOS43MywzOS43MywwLDAsMS0xNS41MiwxNSw0NS4xNSw0NS4xNSwwLDAsMS0yMS44Niw1LjNBNDMuNTUsNDMuNTUsMCwwLDEsMjY2LjY2LDEwMi40OVptMzIuMTYtMTUuNjdhMjAuMDcsMjAuMDcsMCwwLDAsOC04LjI4LDI3LjkyLDI3LjkyLDAsMCwwLDMtMTMuNDRxMC0xMS43Ny02LjItMTguMTNhMjAuMzYsMjAuMzYsMCwwLDAtMTUuMTUtNi4zNCwxOS44NSwxOS44NSwwLDAsMC0xNSw2LjM0cS02LDYuMzUtNi4wNSwxOC4xM3Q1LjksMTguMTRhMTkuNCwxOS40LDAsMCwwLDE0Ljg1LDYuMzRBMjEuNzYsMjEuNzYsMCwwLDAsMjk4LjgyLDg2LjgyWiIvPjxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTM0NC4yNyw0Mi41N2EzNi43NCwzNi43NCwwLDAsMSwxMy41OC0xNC45M0EzNi4xOSwzNi4xOSwwLDAsMSwzNzcsMjIuNDJhMzMuNzQsMzMuNzQsMCwwLDEsMTYuMTksMy43MywzMi44MSwzMi44MSwwLDAsMSwxMS4xMiw5LjRWMjMuNzZoMjEuMDV2ODIuNjhINDA0LjM0Vjk0LjM2QTMxLDMxLDAsMCwxLDM5My4yMiwxMDRhMzQsMzQsMCwwLDEtMTYuMzQsMy44MSwzNS4wOSwzNS4wOSwwLDAsMS0xOS01LjM4LDM3LjY3LDM3LjY3LDAsMCwxLTEzLjU4LTE1LjE0LDQ4LjcxLDQ4LjcxLDAsMCwxLTUtMjIuNDZBNDcuOSw0Ny45LDAsMCwxLDM0NC4yNyw0Mi41N1ptNTcuMDksOS40OGEyMSwyMSwwLDAsMC04LjA2LTguMzYsMjEuNTUsMjEuNTUsMCwwLDAtMTAuOS0yLjkxLDIxLjEzLDIxLjEzLDAsMCwwLTEwLjc0LDIuODMsMjEuNDIsMjEuNDIsMCwwLDAtOCw4LjI5LDI1LjkyLDI1LjkyLDAsMCwwLTMuMDUsMTIuOTEsMjYuNzcsMjYuNzcsMCwwLDAsMy4wNSwxMywyMi4wOSwyMi4wOSwwLDAsMCw4LjA2LDguNTksMjAuNDgsMjAuNDgsMCwwLDAsMTAuNjcsMywyMS41NSwyMS41NSwwLDAsMCwxMC45LTIuOTEsMjEsMjEsMCwwLDAsOC4wNi04LjM2LDI2Ljc4LDI2Ljc4LDAsMCwwLDMtMTMuMDZBMjYuNzcsMjYuNzcsMCwwLDAsNDAxLjM2LDUyLjA1WiIvPjxwYXRoIGNsYXNzPSJjbHMtMiIgZD0iTTUxMi40NCwzMS45cTkuMDksOS4zMyw5LjEsMjZ2NDguNUg1MDAuNjVWNjAuNzhxMC05Ljg2LTQuOTMtMTUuMTV0LTEzLjQzLTUuM3EtOC42NiwwLTEzLjY2LDUuM3QtNSwxNS4xNXY0NS42NmgtMjAuOVYyMy43NmgyMC45djEwLjNhMjguODIsMjguODIsMCwwLDEsMTAuNjctOC40MywzMywzMywwLDAsMSwxNC4yNS0zLjA2UTUwMy4zNCwyMi41Nyw1MTIuNDQsMzEuOVoiLz48cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik01OTcuNTksMi4yN1YxOS4xNEg1NjkuODN2ODcuM0g1NDguOTRWMTkuMTRINTIxLjE4VjIuMjdaIi8+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNNTk4LDQyLjU3YTM2LjY3LDM2LjY3LDAsMCwxLDEzLjU4LTE0LjkzLDM2LjE3LDM2LjE3LDAsMCwxLDE5LjE4LTUuMjIsMzMuNzcsMzMuNzcsMCwwLDEsMTYuMTksMy43MywzMi45LDMyLjksMCwwLDEsMTEuMTIsOS40VjIzLjc2aDIxdjgyLjY4aC0yMVY5NC4zNkEzMS4xOCwzMS4xOCwwLDAsMSw2NDYuOTMsMTA0YTM0LDM0LDAsMCwxLTE2LjM0LDMuODEsMzUuMDYsMzUuMDYsMCwwLDEtMTktNS4zOEEzNy41OSwzNy41OSwwLDAsMSw1OTgsODcuMjdhNDguNTksNDguNTksMCwwLDEtNS0yMi40NkE0Ny43OCw0Ny43OCwwLDAsMSw1OTgsNDIuNTdabTU3LjA4LDkuNDhhMjEsMjEsMCwwLDAtOC04LjM2LDIxLjU4LDIxLjU4LDAsMCwwLTEwLjktMi45MSwyMS4xMywyMS4xMywwLDAsMC0xMC43NCwyLjgzLDIxLjM1LDIxLjM1LDAsMCwwLTgsOC4yOSwyNS45MywyNS45MywwLDAsMC0zLjA2LDEyLjkxLDI2Ljc3LDI2Ljc3LDAsMCwwLDMuMDYsMTMsMjIuMDksMjIuMDksMCwwLDAsOC4wNiw4LjU5LDIwLjQ2LDIwLjQ2LDAsMCwwLDEwLjY3LDNBMjEuNTgsMjEuNTgsMCwwLDAsNjQ3LDg2LjUyYTIxLDIxLDAsMCwwLDgtOC4zNiwyNi43OCwyNi43OCwwLDAsMCwzLTEzLjA2QTI2Ljc3LDI2Ljc3LDAsMCwwLDY1NS4wNiw1Mi4wNVoiLz48cGF0aCBjbGFzcz0iY2xzLTEiIGQ9Ik03MjguNDYsMjYuMjNhMzMuNjQsMzMuNjQsMCwwLDEsMTYuMTktMy44MSwzNi4xNywzNi4xNywwLDAsMSwxOS4xOCw1LjIyLDM2LjgzLDM2LjgzLDAsMCwxLDEzLjU4LDE0Ljg1LDQ3LjgxLDQ3LjgxLDAsMCwxLDUsMjIuMzIsNDguNzEsNDguNzEsMCwwLDEtNSwyMi40NiwzNy41OSwzNy41OSwwLDAsMS0xMy41OCwxNS4xNCwzNS4zMywzNS4zMywwLDAsMS0xOS4xOCw1LjM4LDMzLjQ0LDMzLjQ0LDAsMCwxLTE2LTMuNzMsMzQuMzgsMzQuMzgsMCwwLDEtMTEuMjctOS40MXY1MS4xOUg2OTYuNDVWMjMuNzZoMjAuODlWMzUuN0EzMS42NSwzMS42NSwwLDAsMSw3MjguNDYsMjYuMjNaTTc1OCw1MS45QTIxLjI0LDIxLjI0LDAsMCwwLDc1MCw0My42MWEyMS41NiwyMS41NiwwLDAsMC0xMC44Mi0yLjgzLDIwLjg2LDIwLjg2LDAsMCwwLTEwLjY3LDIuOTEsMjEuNjMsMjEuNjMsMCwwLDAtOC4wNiw4LjQzLDI2LjM2LDI2LjM2LDAsMCwwLTMuMDYsMTMsMjYuMzcsMjYuMzcsMCwwLDAsMy4wNiwxMywyMS42MywyMS42MywwLDAsMCw4LjA2LDguNDNBMjEuMDgsMjEuMDgsMCwwLDAsNzUwLDg2LjQ1LDIyLjIsMjIuMiwwLDAsMCw3NTgsNzcuOTRhMjYuNjksMjYuNjksMCwwLDAsMy4wNi0xMy4xM0EyNS45MywyNS45MywwLDAsMCw3NTgsNTEuOVoiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik0yMDIuNywxMjZWMTI5aC0xMnY5LjMyaDkuNzZ2Mi45NGgtOS43NnYxMi40NmgtMy42MVYxMjZaIi8+PHBhdGggY2xhc3M9ImNscy0zIiBkPSJNMjA5LjM0LDEzNi44NmE5LjcxLDkuNzEsMCwwLDEsMy43MS0zLjkxLDEwLjE5LDEwLjE5LDAsMCwxLDUuMjUtMS4zOCw5LjYzLDkuNjMsMCwwLDEsNSwxLjIzLDguMTgsOC4xOCwwLDAsMSwzLjE0LDMuMDl2LTRoMy42NXYyMS43NUgyMjYuNHYtNGE4LjU5LDguNTksMCwwLDEtMy4yLDMuMTYsOS41Nyw5LjU3LDAsMCwxLTQuOTQsMS4yNSw5LjgzLDkuODMsMCwwLDEtOC45Mi01LjQ0LDEyLjQ4LDEyLjQ4LDAsMCwxLTEuMzUtNS44N0ExMi4yMiwxMi4yMiwwLDAsMSwyMDkuMzQsMTM2Ljg2Wm0xNi4wNywxLjYxYTcuMDksNy4wOSwwLDAsMC0yLjY4LTIuOCw3LjYxLDcuNjEsMCwwLDAtNy40LDAsNi45Myw2LjkzLDAsMCwwLTIuNjYsMi43OCw4Ljg4LDguODgsMCwwLDAtMSw0LjI5LDkuMDYsOS4wNiwwLDAsMCwxLDQuMzQsNyw3LDAsMCwwLDIuNjYsMi44Miw3LjE2LDcuMTYsMCwwLDAsMy42OSwxLDcuMjksNy4yOSwwLDAsMCwzLjcxLTEsNy4wOCw3LjA4LDAsMCwwLDIuNjgtMi44Miw5LjgxLDkuODEsMCwwLDAsMC04LjU5WiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTI0Mi40MiwxNTMuMTdhNy41Miw3LjUyLDAsMCwxLTMuMTMtMi4zOCw2LjMzLDYuMzMsMCwwLDEtMS4yNy0zLjUxaDMuNzNhMy42NCwzLjY0LDAsMCwwLDEuNTMsMi42Niw1Ljc3LDUuNzcsMCwwLDAsMy41OSwxLDUuMjEsNS4yMSwwLDAsMCwzLjI1LS45MSwyLjgsMi44LDAsMCwwLDEuMTktMi4zLDIuMjcsMi4yNywwLDAsMC0xLjI3LTIuMTMsMTguMSwxOC4xLDAsMCwwLTMuOTMtMS4zNywyOC4yMiwyOC4yMiwwLDAsMS00LTEuMjksNyw3LDAsMCwxLTIuNjItMS45NCw1LjUsNS41LDAsMCwxLS4wOS02LjQ1LDYuNzYsNi43NiwwLDAsMSwyLjgxLTIuMiwxMC4yMSwxMC4yMSwwLDAsMSw0LjE3LS44MSw4LjkyLDguOTIsMCwwLDEsNS44MywxLjgyLDYuNTgsNi41OCwwLDAsMSwyLjM4LDVIMjUxYTMuNzIsMy43MiwwLDAsMC0xLjM3LTIuNzQsNi4wNiw2LjA2LDAsMCwwLTYuNDQtLjIsMi42LDIuNiwwLDAsMC0xLjE2LDIuMTksMi40LDIuNCwwLDAsMCwuNywxLjc2LDQuNzgsNC43OCwwLDAsMCwxLjc1LDEuMTFjLjcuMjgsMS42Ny41OSwyLjkxLjk0YTMxLDMxLDAsMCwxLDMuODEsMS4yNSw2LjYzLDYuNjMsMCwwLDEsMi41MiwxLjg0LDQuOTEsNC45MSwwLDAsMSwxLjA5LDMuMjIsNS40OSw1LjQ5LDAsMCwxLTEsMy4yMSw2LjU3LDYuNTcsMCwwLDEtMi44LDIuMjQsMTAsMTAsMCwwLDEtNC4xNC44MkExMS4yNCwxMS4yNCwwLDAsMSwyNDIuNDIsMTUzLjE3WiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTI2Ny4yNSwxMzQuOXYxMi44MmEzLjA2LDMuMDYsMCwwLDAsLjY4LDIuMjQsMy4zNiwzLjM2LDAsMCwwLDIuMzQuNjVoMi42NnYzLjA2aC0zLjI2YTYuNDUsNi40NSwwLDAsMS00LjUyLTEuMzljLTEtLjkzLTEuNTEtMi40NS0xLjUxLTQuNTZWMTM0LjloLTIuODF2LTNoMi44MXYtNS40N2gzLjYxdjUuNDdoNS42OHYzWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTI4NC41OCwxNDkuMjJ2NC40NWgtNC4zMnYtNC40NVoiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik0zMjIuOTEsMTI2VjEyOWgtMTJ2OS4zMmg5Ljc2djIuOTRoLTkuNzZ2MTIuNDZoLTMuNjFWMTI2WiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTMzMy42MywxMjQuM3YyOS4zN0gzMzBWMTI0LjNaIi8+PHBhdGggY2xhc3M9ImNscy0zIiBkPSJNMzYyLjczLDE0NC4xNUgzNDUuMzVhNi44NCw2Ljg0LDAsMCwwLDcuMDYsNi44Miw2LjY1LDYuNjUsMCwwLDAsMy45MS0xLjA5LDUuODEsNS44MSwwLDAsMCwyLjIxLTIuOTJoMy44OGE5LjMyLDkuMzIsMCwwLDEtMy40OSw1LjEsMTAuNTIsMTAuNTIsMCwwLDEtNi41MSwyLDExLDExLDAsMCwxLTUuNTMtMS4zOSw5Ljg0LDkuODQsMCwwLDEtMy44My0zLjk1LDEyLjIsMTIuMiwwLDAsMS0xLjM5LTUuOTMsMTIuNDYsMTIuNDYsMCwwLDEsMS4zNS01LjkyLDkuNTIsOS41MiwwLDAsMSwzLjc5LTMuOTEsMTEuMzEsMTEuMzEsMCwwLDEsNS42MS0xLjM2LDEwLjkxLDEwLjkxLDAsMCwxLDUuNDgsMS4zNSw5LjIzLDkuMjMsMCwwLDEsMy42NywzLjcxLDEwLjkyLDEwLjkyLDAsMCwxLDEuMjksNS4zM0EyMS4yMywyMS4yMywwLDAsMSwzNjIuNzMsMTQ0LjE1Wm0tNC41Mi02LjQ5YTUuOTQsNS45NCwwLDAsMC0yLjQ4LTIuMjYsNy42OSw3LjY5LDAsMCwwLTMuNDctLjc4LDYuNjksNi42OSwwLDAsMC00LjY3LDEuNzUsNy4xNCw3LjE0LDAsMCwwLTIuMiw0Ljg0aDEzLjczQTYuNjMsNi42MywwLDAsMCwzNTguMjEsMTM3LjY2WiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTM4MS42LDE1My42N2wtNS4xNi04LjEtNSw4LjFoLTMuNzdsNy0xMC43OS03LTExaDQuMDlMMzc3LDE0MGw0LjkyLTguMDZoMy43N2wtNywxMC43Niw3LDExWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTM5Mi41NCwxMjcuNjhhMi4zNywyLjM3LDAsMCwxLS43MS0xLjc1LDIuMzMsMi4zMywwLDAsMSwuNzEtMS43NCwyLjM4LDIuMzgsMCwwLDEsMS43NS0uNzIsMi4yNCwyLjI0LDAsMCwxLDEuNjguNzIsMi40LDIuNCwwLDAsMSwuNywxLjc0LDIuNDQsMi40NCwwLDAsMS0uNywxLjc1LDIuMjcsMi4yNywwLDAsMS0xLjY4LjcxQTIuNDEsMi40MSwwLDAsMSwzOTIuNTQsMTI3LjY4Wm0zLjQ5LDQuMjR2MjEu%0ANzVoLTMuNjFWMTMxLjkyWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTQxMi4wOSwxMzIuOGE5LjYyLDkuNjIsMCwwLDEsNC44OC0xLjIzLDEwLjIyLDEwLjIyLDAsMCwxLDUuMjgsMS4zOCw5Ljg3LDkuODcsMCwwLDEsMy42OSwzLjkxLDEyLjMzLDEyLjMzLDAsMCwxLDEuMzUsNS44NiwxMi42LDEyLjYsMCwwLDEtMS4zNSw1Ljg3LDkuODksOS44OSwwLDAsMS05LDUuNDRBOS42Niw5LjY2LDAsMCwxLDQxMiwxNTIuOGE4LjQsOC40LDAsMCwxLTMuMi0zLjE0djRoLTMuNjFWMTI0LjNoMy42MVYxMzZBOC41LDguNSwwLDAsMSw0MTIuMDksMTMyLjhabTEwLjUxLDUuNjNhNi44NSw2Ljg1LDAsMCwwLTIuNjgtMi43OCw3LjQ1LDcuNDUsMCwwLDAtMy43MS0xLDcuMzIsNy4zMiwwLDAsMC0zLjY3LDEsNy4yLDcuMiwwLDAsMC0yLjcsMi44Miw5LjU5LDkuNTksMCwwLDAsMCw4LjU3LDcuMzUsNy4zNSwwLDAsMCwxMC4wOCwyLjgyLDcsNywwLDAsMCwyLjY4LTIuODIsOS4wNiw5LjA2LDAsMCwwLDEtNC4zNEE4Ljg4LDguODgsMCwwLDAsNDIyLjYsMTM4LjQzWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTQzOC44OCwxMjQuM3YyOS4zN2gtMy42MVYxMjQuM1oiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik00NjgsMTQ0LjE1SDQ1MC42YTYuODQsNi44NCwwLDAsMCw3LjA2LDYuODIsNi42NSw2LjY1LDAsMCwwLDMuOTEtMS4wOSw1LjgxLDUuODEsMCwwLDAsMi4yMS0yLjkyaDMuODhhOS4zMiw5LjMyLDAsMCwxLTMuNDksNS4xLDEwLjUyLDEwLjUyLDAsMCwxLTYuNTEsMiwxMSwxMSwwLDAsMS01LjUzLTEuMzksOS45MSw5LjkxLDAsMCwxLTMuODMtMy45NSwxMi4yLDEyLjIsMCwwLDEtMS4zOS01LjkzLDEyLjQ2LDEyLjQ2LDAsMCwxLDEuMzUtNS45Miw5LjUyLDkuNTIsMCwwLDEsMy43OS0zLjkxLDExLjMxLDExLjMxLDAsMCwxLDUuNjEtMS4zNiwxMC45MSwxMC45MSwwLDAsMSw1LjQ4LDEuMzUsOS4yMyw5LjIzLDAsMCwxLDMuNjcsMy43MUExMC45MiwxMC45MiwwLDAsMSw0NjguMSwxNDIsMjEuMjMsMjEuMjMsMCwwLDEsNDY4LDE0NC4xNVptLTQuNTItNi40OUE1Ljk0LDUuOTQsMCwwLDAsNDYxLDEzNS40YTcuNjksNy42OSwwLDAsMC0zLjQ3LS43OCw2LjY5LDYuNjksMCwwLDAtNC42NywxLjc1LDcuMTQsNy4xNCwwLDAsMC0yLjIsNC44NGgxMy43M0E2LjYzLDYuNjMsMCwwLDAsNDYzLjQ2LDEzNy42NloiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik00NzguOTMsMTQ5LjIydjQuNDVINDc0LjZ2LTQuNDVaIi8+PHBhdGggY2xhc3M9ImNscy0zIiBkPSJNNTE3LjI2LDEyNlYxMjloLTEydjkuMzJINTE1djIuOTRoLTkuNzd2MTIuNDZoLTMuNjFWMTI2WiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTUyOS41MiwxMzIuNTZhOC4zNSw4LjM1LDAsMCwxLDQuMy0xdjMuNzNoLTFxLTYuMDcsMC02LjA3LDYuNTh2MTEuODNoLTMuNjFWMTMxLjkyaDMuNjF2My41M0E2LjksNi45LDAsMCwxLDUyOS41MiwxMzIuNTZaIi8+PHBhdGggY2xhc3M9ImNscy0zIiBkPSJNNTQxLDEyNy42OGEyLjQxLDIuNDEsMCwwLDEtLjcxLTEuNzUsMi40NCwyLjQ0LDAsMCwxLDIuNDYtMi40NiwyLjI4LDIuMjgsMCwwLDEsMS42OS43MiwyLjM5LDIuMzksMCwwLDEsLjY5LDEuNzQsMi40MywyLjQzLDAsMCwxLS42OSwxLjc1LDIuMywyLjMsMCwwLDEtMS42OS43MUEyLjQxLDIuNDEsMCwwLDEsNTQxLDEyNy42OFptMy41LDQuMjR2MjEuNzVoLTMuNjFWMTMxLjkyWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTU3My4zNiwxNDQuMTVINTU2QTYuODQsNi44NCwwLDAsMCw1NjMsMTUxYTYuNjUsNi42NSwwLDAsMCwzLjkxLTEuMDksNS43Myw1LjczLDAsMCwwLDIuMi0yLjkySDU3M2E5LjMyLDkuMzIsMCwwLDEtMy40OSw1LjEsMTAuNTYsMTAuNTYsMCwwLDEtNi41MSwyLDExLDExLDAsMCwxLTUuNTMtMS4zOSw5Ljg0LDkuODQsMCwwLDEtMy44My0zLjk1LDEyLjIsMTIuMiwwLDAsMS0xLjM5LTUuOTMsMTIuNDYsMTIuNDYsMCwwLDEsMS4zNS01LjkyLDkuNTIsOS41MiwwLDAsMSwzLjc5LTMuOTEsMTEuMjgsMTEuMjgsMCwwLDEsNS42MS0xLjM2LDEwLjkxLDEwLjkxLDAsMCwxLDUuNDgsMS4zNSw5LjIzLDkuMjMsMCwwLDEsMy42NywzLjcxLDEwLjkyLDEwLjkyLDAsMCwxLDEuMjksNS4zM0EyMS4yMywyMS4yMywwLDAsMSw1NzMuMzYsMTQ0LjE1Wm0tNC41Mi02LjQ5YTUuOTQsNS45NCwwLDAsMC0yLjQ4LTIuMjYsNy43Myw3LjczLDAsMCwwLTMuNDgtLjc4LDYuNjcsNi42NywwLDAsMC00LjY2LDEuNzUsNy4xLDcuMSwwLDAsMC0yLjIsNC44NGgxMy43M0E2LjcyLDYuNzIsMCwwLDAsNTY4Ljg0LDEzNy42NloiLz48cGF0aCBjbGFzcz0iY2xzLTMiIGQ9Ik01OTguNDQsMTMzLjkzcTIuNDYsMi40LDIuNDYsNi45MnYxMi44MmgtMy41N3YtMTIuM2E3LDcsMCwwLDAtMS42My01LDUuOCw1LjgsMCwwLDAtNC40NC0xLjczLDUuOTMsNS45MywwLDAsMC00LjU0LDEuNzksNy4yNyw3LjI3LDAsMCwwLTEuNjksNS4ydjEyaC0zLjYxVjEzMS45Mkg1ODVWMTM1YTcuMTcsNy4xNywwLDAsMSwyLjkyLTIuNTgsOSw5LDAsMCwxLDQuMDYtLjkxQTguODEsOC44MSwwLDAsMSw1OTguNDQsMTMzLjkzWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTYxMCwxMzYuODZhOS43OCw5Ljc4LDAsMCwxLDMuNzEtMy45MSwxMC4zLDEwLjMsMCwwLDEsNS4zLTEuMzgsOS44NSw5Ljg1LDAsMCwxLDQuNzIsMS4xNyw4LjQ3LDguNDcsMCwwLDEsMy4zMywzLjA3VjEyNC4zaDMuNjV2MjkuMzdINjI3di00LjA5YTguNDYsOC40NiwwLDAsMS0zLjE3LDMuMiw5LjUyLDkuNTIsMCwwLDEtNC45MiwxLjI1LDkuODksOS44OSwwLDAsMS05LTUuNDQsMTIuNDgsMTIuNDgsMCwwLDEtMS4zNS01Ljg3QTEyLjIyLDEyLjIyLDAsMCwxLDYxMCwxMzYuODZaTTYyNiwxMzguNDdhNyw3LDAsMCwwLTIuNjgtMi44LDcuNjEsNy42MSwwLDAsMC03LjQsMCw2LjkzLDYuOTMsMCwwLDAtMi42NiwyLjc4LDguODgsOC44OCwwLDAsMC0xLDQuMjksOS4wNiw5LjA2LDAsMCwwLDEsNC4zNCw3LDcsMCwwLDAsMi42NiwyLjgyLDcuMTYsNy4xNiwwLDAsMCwzLjY5LDEsNy4yNiw3LjI2LDAsMCwwLDMuNzEtMSw3LDcsMCwwLDAsMi42OC0yLjgyLDkuODEsOS44MSwwLDAsMCwwLTguNTlaIi8+PHBhdGggY2xhc3M9ImNscy0zIiBkPSJNNjQzLjQyLDEyNC4zdjI5LjM3aC0zLjYxVjEyNC4zWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTY3MS40NiwxMzEuOTJsLTEzLjEsMzJoLTMuNzNsNC4yOS0xMC40OC04Ljc3LTIxLjUxaDRMNjYxLDE0OS41NGw2Ljc1LTE3LjYyWiIvPjxwYXRoIGNsYXNzPSJjbHMtMyIgZD0iTTY3OS4xMSwxNDkuMjJ2NC40NWgtNC4zMnYtNC40NVoiLz48L3N2Zz4=", width=300)
st.sidebar.title("About LoanTap")
st.sidebar.info("""
**LoanTap** is a leading fintech company specializing in personal loan solutions.

### 📌 Problem Statement
We aim to build a machine learning model that predicts **loan approval status** based on applicant details.

### 🧠 Model Details
- **Algorithm:** Logistic Regression
- **Features:** Loan Amount, Interest Rate, DTI, Credit Grade, Income, etc.
- **Target:** Loan Status (Approved/Rejected)
""")

# ---- Header with Company Banner ----
st.markdown(
    """
    <div style="text-align: center;">
        <img src="company_banner.png" width="80%">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center; color: #004AAD;'>🏦 Loan Approval Prediction</h1>"
    "<p style='text-align: center; font-size: 18px;'>💡 Get an instant prediction on your loan approval status!</p>",
    unsafe_allow_html=True
)

# ---- Layout with Columns ----
col1, col2 = st.columns([1, 1])

# ---- Left Column: User Inputs ----
with col1:
    st.markdown("### 📋 **Enter Loan Details**")
    loan_amnt = st.number_input("💰 **Loan Amount (USD)**", min_value=500, max_value=50000, step=500, help="Enter the amount you wish to borrow.")
    term = st.selectbox("📆 **Loan Term (Months)**", term_options, help="Select the duration of the loan.")
    int_rate = st.number_input("📊 **Interest Rate (%)**", min_value=1.0, max_value=30.0, step=0.1, help="Enter the interest rate for your loan.")
    dti = st.number_input("📉 **Debt-to-Income Ratio (DTI)**", min_value=0.0, max_value=50.0, step=0.1, help="DTI measures your monthly debt payments against your income.")
    purpose = st.selectbox("🎯 **Purpose of Loan**", purpose_options, help="Select the purpose of your loan.")
    verification_status = st.selectbox("✅ **Verification Status**", verification_status_options, help="Indicates whether your income is verified.")
    grade = st.selectbox("🏅 **Credit Grade**", grade_options, help="Your creditworthiness level assigned by the lender.")
    annual_inc = st.number_input("💵 **Annual Income (USD)**", min_value=10000, max_value=500000, step=1000, help="Enter your yearly income before tax.")
    home_ownership = st.selectbox("🏠 **Home Ownership**", home_ownership_options, help="Your current housing situation.")
    Credit_History_Years = st.number_input("📜 **Credit History (Years)**", min_value=0, max_value=50, step=1, help="Years since your first credit account.")

# Prepare input data
input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt], 'term': [term], 'int_rate': [int_rate], 'dti': [dti],
    'purpose': [purpose], 'verification_status': [verification_status], 'grade': [grade],
    'annual_inc': [annual_inc], 'home_ownership': [home_ownership], 'Credit_History_Years': [Credit_History_Years]
})

# One-Hot Encoding for Categorical Variables
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=feature_columns, fill_value=0)

# Scale numerical values
input_data_scaled = scaler.transform(input_data)

# Prediction function
def predict_loan_status(input_data_scaled):
    prediction = model.predict(input_data_scaled)
    probability = model.predict_proba(input_data_scaled)[0][1]
    return ('Approved ✅' if prediction[0] == 1 else 'Rejected ❌', probability)

# ---- Right Column: Loan Summary & Prediction ----
with col2:
    st.markdown("### 📊 **Loan Summary**")
    st.info(f"""
    - **Loan Amount:** ${loan_amnt}
    - **Term:** {term} months
    - **Interest Rate:** {int_rate}%
    - **Debt-to-Income Ratio:** {dti}%
    - **Purpose:** {purpose}
    - **Verification Status:** {verification_status}
    - **Credit Grade:** {grade}
    - **Annual Income:** ${annual_inc}
    - **Home Ownership:** {home_ownership}
    - **Credit History:** {Credit_History_Years} years
    """)

    if st.button("🔍 **Predict Loan Approval**"):
        result, probability = predict_loan_status(input_data_scaled)
        if result == "Approved ✅":
            st.success(f"🎉 **Congratulations! Your loan is likely to be {result}**")
        else:
            st.error(f"⚠️ **Unfortunately, your loan is likely to be {result}**")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center;">
        <p>🚀 Created by <b>Aman Shrivastava</b></p>
        <p>📧 Contact: <a href="mailto:amanshrivastava26266@gmail.com">amanshrivastava26266@gmail.com</a></p>
        <p>🔗 <a href="https://www.linkedin.com/in/aman0802/" target="_blank">LinkedIn</a> | 
        <a href="https://github.com/0825aman" target="_blank">GitHub</a> | 
        <a href="https://www.kaggle.com/the0aman0shrivastava" target="_blank">Kaggle</a></p>
    </div>
""", unsafe_allow_html=True)
