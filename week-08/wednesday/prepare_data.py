import pandas as pd
import numpy as np
import os

def generate_social_media_data():
    reddit_path = r"C:\Users\Avi\Downloads\archive (1)\Reddit_Data.csv"
    twitter_path = r"C:\Users\Avi\Downloads\archive (1)\Twitter_Data.csv"
    
    print("Loading Reddit")
    df_reddit = pd.read_csv(reddit_path).dropna()
    print("Loading Twitter")
    df_twitter = pd.read_csv(twitter_path).dropna()
    
    df_reddit = df_reddit.rename(columns={'clean_comment': 'text', 'category': 'sentiment'})
    df_reddit['platform'] = 'Reddit'
    
    df_twitter = df_twitter.rename(columns={'clean_text': 'text', 'category': 'sentiment'})
    df_twitter['platform'] = 'Twitter'
    
    df_combined = pd.concat([df_reddit, df_twitter], ignore_index=True)
    df_sampled = df_combined.sample(n=3000, random_state=42).reset_index(drop=True)
    
    np.random.seed(42)
    languages = ['en', 'en', 'en', 'en', 'es', 'fr', 'hi']
    df_sampled['language'] = np.random.choice(languages, 3000)
    
    topics = ['politics', 'sports', 'world', 'technology', 'finance']
    df_sampled['topic'] = np.random.choice(topics, 3000)
    
    hate_probs = np.where(df_sampled['sentiment'] == -1, 0.15, 0.01) 
    df_sampled['hate_speech_flag'] = np.random.binomial(1, hate_probs)
    
    spam_probs = np.where(df_sampled['topic'] == 'finance', 0.2, 0.05)
    df_sampled['spam_flag'] = np.random.binomial(1, spam_probs)
    
    for col in ['language', 'topic']:
        mask = np.random.rand(3000) < 0.05
        df_sampled.loc[mask, col] = np.nan
        
    df_sampled.to_csv("social_media_posts.csv", index=False)
    print("Generated social_media_posts.csv")

if __name__ == "__main__":
    generate_social_media_data()
