# Import necessary libraries
import streamlit as st
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Web scraping function to get news article links
def scrape_news_links(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    links = set()  # Using a set to avoid duplicate links
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and 'article' in href:  # Adjust this condition based on the structure of the links
            links.add(href)
    return list(links)  # Convert set back to list for clustering

# Clustering function
def cluster_links(links):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(links)

    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    clustered_links = {i: [] for i in range(true_k)}  # Initialize an empty list for each cluster
    for i in range(len(links)):
        clustered_links[model.labels_[i]].append(links[i])

    return clustered_links

# Main function to run the Streamlit web app
def main():
    st.title('Clustered News Article Links')

    news_url = st.text_input('Enter URL of the news website to scrape:', 'https://www.standardmedia.co.ke/')

    if st.button('Scrape Links and Cluster'):
        news_links = scrape_news_links(news_url)
        clustered_links = cluster_links(news_links)

        st.subheader('Clustered News Article Links')
        for cluster_id, links in clustered_links.items():
            st.write(f'Cluster {cluster_id + 1} ({len(links)} links)')
            for link in links:
                st.write(link)

if __name__ == '__main__':
    main()
