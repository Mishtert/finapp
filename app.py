import streamlit as st
import tech_analysis
import stock_sentiment
import optimization
import optimization2
import stock_screener
import cx_segmentation
import anomaly_detection
import rx_mgt


# st.page_title = 'Python & Machine Learning Financial Applications'


max_width = 5000,
padding_top = 1,
padding_right = 1
padding_left = 1
padding_bottom = 1
COLOR = 'white',
BACKGROUND_COLOR = 'white'

hide_footer_style = """
<style>
.reportview-container .main footer {visibility: hidden;}    
"""
st.markdown(hide_footer_style, unsafe_allow_html=True)

html = """
  <style>
    .reportview-container {
      flex-direction: row-reverse;
    }

    header > .toolbar {
      flex-direction: row-reverse;
      left: 1rem;
      right: auto;
    }

    .sidebar .sidebar-collapse-control,
    .sidebar.--collapsed .sidebar-collapse-control {
      left: auto;
      right: 0.3rem;
    }

    .sidebar .sidebar-content {
      transition: margin-right .3s, box-shadow .3s;
    }

    .sidebar.--collapsed .sidebar-content {
      margin-left: auto;
      margin-right: -21rem;
    }

    @media (max-width: 991.98px) {
      .sidebar .sidebar-content {
        margin-left: auto;
      }
      .sidebar.sidebar-width{
        8px
      }
    }
  </style>
"""
st.markdown(html, unsafe_allow_html=True)

PAGES = {
    'Customer Segmentation (Credit Card) ': cx_segmentation,
    'Fraud Detection': anomaly_detection,
    'Stock Screening (Shortlisting) ': stock_screener,
    'Portfolio Optimization ': optimization2,
    'Portfolio Reallocation with Cost Objective': optimization,
    'Value @ Risk': rx_mgt,
    # "Technical Analysis": tech_analysis,

    # "Stock News": stock_sentiment,

}


def main():

    # enable_fullscreen_content()
    # st.right(sidebar=True, width=200)
    st.header('Python & Machine Learning Financial Applications')
    # st.header('Python & Machine Learning Financial Applications')
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    st.sidebar.subheader("About App")
    st.sidebar.text("Showcase of financial applications")

    st.sidebar.subheader("Design & Development")
    st.sidebar.text("Mishtert T")


    with st.spinner(f"Loading {selection} Page ..."):
        page.write()  # each page has a write function


if __name__ == "__main__":
    main()
