import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("medical_examination.csv")

df["overweight"] = ((df.weight / ((df.height / 100) ** 2)) > 25).astype(int)

df.cholesterol = (df.cholesterol > 1).astype(int)
df.gluc = (df.gluc > 1).astype(int)


# 4
def draw_cat_plot():
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight",
        ],
    )

    df_cat = (
        df_cat.groupby(["cardio", "variable"]).value_counts().reset_index(name="total")
    )

    plot = sns.catplot(
        df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
    )

    fig = plot.fig

    fig.savefig("catplot.png")
    return fig


# 10
def draw_heat_map():
    df_heat = df.loc[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    corr = df_heat.corr()
    mask = np.triu(np.ones(corr.shape))

    fig, ax = plt.subplots()

    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        fmt=".1f",
    )

    fig.savefig("heatmap.png")
    return fig


draw_heat_map()
