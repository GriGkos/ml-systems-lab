"""Feature engineering for the application-only Home Credit scoring model."""

import numpy as np
import pandas as pd


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely, preserving missing values instead of infinities."""
    return numerator / denominator.replace(0, np.nan)


def enrich_application(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the deterministic application features used in the source project."""
    result = frame.copy()
    result["APP__EMPLOYMENT_ANOMALY"] = result["DAYS_EMPLOYED"].eq(365243).astype("int8")
    result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(365243, np.nan)

    result["APP__AGE_YEARS"] = -result["DAYS_BIRTH"] / 365.25
    result["APP__EMPLOYMENT_YEARS"] = -result["DAYS_EMPLOYED"] / 365.25
    result["APP__REGISTRATION_YEARS"] = -result["DAYS_REGISTRATION"] / 365.25
    result["APP__ID_PUBLISH_YEARS"] = -result["DAYS_ID_PUBLISH"] / 365.25
    result["APP__PHONE_CHANGE_YEARS"] = -result["DAYS_LAST_PHONE_CHANGE"] / 365.25
    result["APP__AGE_AT_EMPLOYMENT_START"] = (
        result["APP__AGE_YEARS"] - result["APP__EMPLOYMENT_YEARS"]
    )

    ratios = {
        "APP__CREDIT_TO_INCOME": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
        "APP__ANNUITY_TO_INCOME": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        "APP__CREDIT_TO_ANNUITY": ("AMT_CREDIT", "AMT_ANNUITY"),
        "APP__PAYMENT_RATE": ("AMT_ANNUITY", "AMT_CREDIT"),
        "APP__GOODS_TO_CREDIT": ("AMT_GOODS_PRICE", "AMT_CREDIT"),
        "APP__CREDIT_PER_PERSON": ("AMT_CREDIT", "CNT_FAM_MEMBERS"),
        "APP__INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
        "APP__CHILD_TO_FAMILY": ("CNT_CHILDREN", "CNT_FAM_MEMBERS"),
        "APP__EMPLOYED_TO_AGE": ("APP__EMPLOYMENT_YEARS", "APP__AGE_YEARS"),
    }
    for name, (numerator, denominator) in ratios.items():
        result[name] = safe_ratio(result[numerator], result[denominator])
    result["APP__INCOME_PER_CHILD"] = safe_ratio(
        result["AMT_INCOME_TOTAL"], result["CNT_CHILDREN"] + 1
    )

    result["APP__V2_CREDIT_GOODS_GAP"] = result["AMT_CREDIT"] - result["AMT_GOODS_PRICE"]
    result["APP__V2_CREDIT_GOODS_GAP_RATIO"] = safe_ratio(
        result["APP__V2_CREDIT_GOODS_GAP"], result["AMT_GOODS_PRICE"]
    )
    result["APP__V2_ANNUITY_TO_GOODS"] = safe_ratio(
        result["AMT_ANNUITY"], result["AMT_GOODS_PRICE"]
    )

    ext_sources = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    result["APP__EXT_SOURCE_MEAN"] = result[ext_sources].mean(axis=1)
    result["APP__EXT_SOURCE_STD"] = result[ext_sources].std(axis=1)
    result["APP__EXT_SOURCE_MIN"] = result[ext_sources].min(axis=1)
    result["APP__EXT_SOURCE_MAX"] = result[ext_sources].max(axis=1)
    result["APP__EXT_SOURCE_RANGE"] = result["APP__EXT_SOURCE_MAX"] - result["APP__EXT_SOURCE_MIN"]
    result["APP__EXT_SOURCE_AVAILABLE"] = result[ext_sources].notna().sum(axis=1).astype("int8")
    result["APP__V2_EXT_1_X_2"] = result["EXT_SOURCE_1"] * result["EXT_SOURCE_2"]
    result["APP__V2_EXT_1_X_3"] = result["EXT_SOURCE_1"] * result["EXT_SOURCE_3"]
    result["APP__V2_EXT_2_X_3"] = result["EXT_SOURCE_2"] * result["EXT_SOURCE_3"]
    result["APP__V2_EXT_PRODUCT"] = result[ext_sources].prod(axis=1, min_count=3)
    result["APP__V2_EXT_MEAN_X_AGE"] = result["APP__EXT_SOURCE_MEAN"] * result["APP__AGE_YEARS"]
    result["APP__V2_EXT_MEAN_X_CREDIT_TO_INCOME"] = (
        result["APP__EXT_SOURCE_MEAN"] * result["APP__CREDIT_TO_INCOME"]
    )

    contact_columns = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ]
    result["APP__CONTACT_FLAGS_SUM"] = result[contact_columns].sum(axis=1)
    result["APP__DOCUMENT_FLAGS_SUM"] = result[
        [column for column in result if column.startswith("FLAG_DOCUMENT_")]
    ].sum(axis=1)
    result["APP__REGION_MISMATCH_SUM"] = result[
        [column for column in result if "_REGION_NOT_" in column or "_CITY_NOT_" in column]
    ].sum(axis=1)
    result["APP__BUREAU_REQUESTS_SUM"] = result[
        [column for column in result if column.startswith("AMT_REQ_CREDIT_BUREAU_")]
    ].sum(axis=1)

    for column in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        result[f"APP__LOG1P__{column}"] = np.log1p(result[column].clip(lower=0))
        result[f"APP__{column}__MISSING"] = result[column].isna().astype("int8")

    return result.replace([np.inf, -np.inf], np.nan)
