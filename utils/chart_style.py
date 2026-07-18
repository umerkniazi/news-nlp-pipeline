SENTIMENT_COLORS = {
    'positive': '#2ecc71',
    'neutral': '#95a5a6',
    'negative': '#e74c3c'
}


def apply_chart_style(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        margin=dict(l=50, r=50, t=60, b=50),
        hovermode="x unified"
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1
    )

    return fig