import streamlit as st
import pandas as pd
import pickle
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# from streamlit.runtime.scriptrunner import rerun


# Shared footer
def footer():
    st.markdown("---")
    st.caption("👨‍💻 Made by Vishal Rai")

# 🎭 Genre ID Map
genre_id_map = {
    28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
    80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
    14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
    9648: 'Mystery', 10749: 'Romance', 878: 'Sci-Fi',
    10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
}

# 🧠 Explain shared tags
def explain_overlap(movie1, movie2):
    try:
        tags1 = set(movies[movies['title'] == movie1]['tags'].values[0].split())
        tags2 = set(movies[movies['title'] == movie2]['tags'].values[0].split())
        common = tags1.intersection(tags2)
        return ', '.join(tag.strip("'\", ") for tag in list(common)[:5])
    except:
        return "N/A"

# 🎬 Extract metadata from row
def get_movie_details(row):
    # Genres
    genres = row['genre_ids']
    if isinstance(genres, str):
        genres = ', '.join([g.strip().capitalize() for g in genres.split(',')])
    else:
        genres = "Unknown"

    # Cast
    cast = row['cast']
    if isinstance(cast, str):
        cast = ', '.join([c.strip().replace('_', ' ').title() for c in cast.split(',')[:3]])
    else:
        cast = "Unknown"

    # Director
    director = row['crew']
    if isinstance(director, str):
        director = director.replace('_', ' ').title()
    else:
        director = "Unknown"

    # Release Year
    year = row['release_date'][:4] if 'release_date' in row and pd.notna(row['release_date']) else "N/A"

    # Rating
    rating = f"{row['vote_average']}/10" if 'vote_average' in row else "N/A"

    return cast, director, genres, rating, year


# 📦 Load Data
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# 🤖 Recommendation Logic
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# -------------------------------
# 🌟 Sidebar Navigation
# -------------------------------
st.sidebar.title("🎥 CineScope")
page = st.sidebar.radio("Go to", ["🎬 Recommendation", "📈 Insights", "🕰️ Movie History", "📚 Fun Facts", "🎮 Quiz Mode"])

# -------------------------------
# 🎬 Page 1: Movie Recommender
# -------------------------------
if page == "🎬 Recommendation":
    st.title('🎬 Smart Movie Recommendation')

    selected_movie_name = st.selectbox(
        'Select a movie to get recommendations:',
        movies['title'].values
    )

    if st.button('Recommend'):
        recommendations = recommend(selected_movie_name)
        st.write("### 🎯 Recommended Movies:")

        cols = st.columns(3)
        for idx, rec_title in enumerate(recommendations):
            try:
                rec_data = movies[movies['title'] == rec_title]
                if rec_data.empty:
                    continue

                row_data = rec_data.iloc[0]

                # ✅ Poster
                poster_url = row_data['poster_path']
                if not poster_url or not str(poster_url).startswith("http"):
                    poster_url = "https://via.placeholder.com/300x450?text=No+Image"

                # ✅ Metadata
                overview = row_data['overview'] if 'overview' in row_data and pd.notna(row_data['overview']) else ""
                cast, director, genres, rating, year = get_movie_details(row_data)
                common_tags = explain_overlap(selected_movie_name, rec_title)



                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="
                        background-color: #1a1a1a;
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        border-radius: 10px;
                        padding: 16px;
                        margin-bottom: 20px;
                        box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
                    ">
                        <img src="{poster_url}" style="width: 100%; border-radius: 8px; margin-bottom: 12px;">
                        <h4 style="color: white; margin-bottom: 8px;">🎬 {rec_title}</h4>
                        <p style="color: #aaa; font-style: italic;">{overview[:150]}...</p>
                        <p><b>📅 Release Year:</b> <span style="color: #ddd;">{year}</span></p>
                        <p><b>⭐ TMDB Rating:</b> <span style="color: #ddd;">{rating}</span></p>
                        <p><b>🎭 Genres:</b> <span style="color: #ddd;">{genres.title()}</span></p>
                        <p><b>👥 Cast:</b> <span style="color: #ddd;">{cast}</span></p>
                        <p><b>🎬 Director:</b> <span style="color: #ddd;">{director}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"❌ Could not process: {rec_title} — Reason: {e}")

    footer()           

# -------------------------------
# Other Pages (Placeholder)
# -------------------------------
elif page == "📈 Insights":
    st.title("📊 Movie Insights Dashboard")
    st.markdown("Explore genre trends, rating patterns, and top movies interactively!")


    # ---- Sidebar Filters ----
    st.sidebar.markdown("### 🎛️ Filters")
    unique_genres = set(g.strip().capitalize() for sub in movies['genre_ids'].dropna().tolist() for g in sub.split(','))
    selected_genre = st.sidebar.selectbox("🎭 Select Genre", sorted(unique_genres))
    rating_range = st.sidebar.slider("⭐ TMDB Rating Range", 0.0, 10.0, (0.0, 10.0), 0.1)

    # --- 1. Genre Distribution (full) ---
    st.subheader("🎭 Genre Distribution (Overall)")
    genre_counter = Counter()
    for g in movies['genre_ids']:
        for genre in g.split(','):
            genre_counter[genre.strip().capitalize()] += 1
    genre_df = pd.DataFrame(genre_counter.items(), columns=["Genre", "Count"]).sort_values("Count", ascending=False)
    st.bar_chart(genre_df.set_index("Genre"))

    # --- 2. Rating Distribution ---
    st.subheader(f"⭐ Rating Distribution in {selected_genre}")
    genre_filtered = movies[movies['genre_ids'].str.contains(selected_genre.lower(), na=False)]
    fig, ax = plt.subplots()
    sns.histplot(genre_filtered['vote_average'], bins=15, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("TMDB Rating")
    ax.set_ylabel("Number of Movies")
    ax.set_title(f"Ratings for {selected_genre} Movies")
    st.pyplot(fig)

    # --- 3. Release Trend for Selected Genre ---
    st.subheader(f"📅 Movie Releases Over Time ({selected_genre})")
    genre_filtered['release_year'] = genre_filtered['release_date'].str[:4]
    release_trend = genre_filtered['release_year'].value_counts().sort_index()
    st.line_chart(release_trend)

    # --- 4. Top Rated Movies in Filtered Range ---
    st.subheader(f"🏆 Top Rated {selected_genre} Movies")
    top_movies = genre_filtered[
        (genre_filtered['vote_average'] >= rating_range[0]) & 
        (genre_filtered['vote_average'] <= rating_range[1])
    ][['title', 'vote_average']].drop_duplicates().sort_values(by='vote_average', ascending=False).head(10)
    
    for index, row in top_movies.iterrows():
        
            st.markdown(f"""
            <div style="
                background-color: #1a1a1a;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 10px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h5 style="color: white; margin: 0;">
                    🎬 {row['title']}
                </h5>
                <p style="color: gold; font-weight: bold; margin: 0;">
                    ⭐ {row['vote_average']}
                </p>
            </div>
            """, unsafe_allow_html=True)


elif page == "🕰️ Movie History":
    st.title("🕰️ Evolution of Cinema")
    st.markdown("Discover how movies evolved over the decades based on your dataset!")

    # Extract year
    movies['release_year'] = movies['release_date'].str[:4]
    movies = movies[movies['release_year'].str.isnumeric()]
    movies['release_year'] = movies['release_year'].astype(int)
    movies['decade'] = (movies['release_year'] // 10) * 10

    # --- 1. Releases Over Decades ---
    st.subheader("🎬 Number of Movies Released by Decade")
    decade_count = movies['decade'].value_counts().sort_index()
    st.bar_chart(decade_count)

    # --- 2. Top Rated Movie per Decade ---
    st.subheader("🏆 Top Rated Movies by Decade")
    top_decade_movies = (
        movies.sort_values(['decade', 'vote_average'], ascending=[True, False])
        .groupby('decade')
        .first()
        .reset_index()[['decade', 'title', 'vote_average']]
    )
    for _, row in top_decade_movies.iterrows():
        st.markdown(f"""
        <div style="background-color: #1a1a1a; padding: 12px; border-radius: 8px; margin-bottom: 10px;">
            <b style="color: gold;">📅 {row['decade']}s</b><br>
            🎬 <span style="color: white;">{row['title']}</span><br>
            ⭐ <span style="color: gold;">Rating: {row['vote_average']}</span>
        </div>
        """, unsafe_allow_html=True)

    # --- 3. Genre Growth Over Time ---
    st.subheader("📈 Genre Popularity Over Decades")
    selected_genre = st.selectbox("🎭 Choose a genre to view its rise:", sorted(set(g.strip().capitalize() for sub in movies['genre_ids'] for g in sub.split(','))))
    genre_trend = movies[movies['genre_ids'].str.contains(selected_genre.lower())].groupby('decade').size()
    st.line_chart(genre_trend)

    # --- 4. Trivia ---
    st.subheader("📚 Historic Trivia")
    oldest_movie = movies.sort_values('release_year').iloc[0]
    st.markdown(f"""
    🏛️ **Oldest Movie:**  
    🎬 {oldest_movie['title']}  
    📅 Year: {oldest_movie['release_year']}  
    ⭐ Rating: {oldest_movie['vote_average']}
    """)


elif page == "📚 Fun Facts":
    st.title("📚 Fun Movie Facts & Records")
    st.markdown("✨ Dive into surprising trivia pulled straight from your movie dataset!")

    # Fun style block function
    def fun_fact_block(emoji, title, content):
        st.markdown(f"""
        <div style="background-color: #1f1f1f; padding: 15px 20px; border-radius: 10px; margin-bottom: 15px;">
            <h5 style="color: gold; margin: 0;">{emoji} {title}</h5>
            <p style="color: #ddd; margin: 5px 0 0;">{content}</p>
        </div>
        """, unsafe_allow_html=True)

    # 1. Highest Rated Movie
    top_movie = movies.loc[movies['vote_average'].idxmax()]
    fun_fact_block("🏆", "Highest Rated Movie", f"**{top_movie['title']}** (⭐ {top_movie['vote_average']}, 📅 {top_movie['release_date'][:4]})")

    # 2. Longest Movie Title
    longest_title = movies.loc[movies['title'].str.len().idxmax()]
    fun_fact_block("🔠", "Longest Movie Title", f"**{longest_title['title']}** with `{len(longest_title['title'])}` characters")

    # 3. Most Frequent Actor
    from collections import Counter
    actor_counter = Counter()
    for cast in movies['cast']:
        for actor in cast.split(','):
            actor_counter[actor.strip().title()] += 1
    most_common_actor, appearances = actor_counter.most_common(1)[0]
    fun_fact_block("🎭", "Most Featured Actor", f"**{most_common_actor}** in `{appearances}` movies")

    # 4. Most Common Genre
    genre_counter = Counter()
    for g in movies['genre_ids']:
        for genre in g.split(','):
            genre_counter[genre.strip().capitalize()] += 1
    top_genre = genre_counter.most_common(1)[0]
    fun_fact_block("🎬", "Most Popular Genre", f"**{top_genre[0]}** appears in `{top_genre[1]}` movies")

    # 5. Year with Most Releases
    movies['release_year'] = movies['release_date'].str[:4]
    year_counts = movies['release_year'].value_counts()
    top_year = year_counts.idxmax()
    fun_fact_block("📅", "Busiest Year in Cinema", f"**{top_year}** with `{year_counts.max()}` releases")

elif page == "🎮 Quiz Mode":
    import random

    st.title("🎮 CineScope Quiz Mode")
    st.markdown("Test your movie brain with questions on rating, cast, directors, and release dates!")

    # Init session state
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = {}
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False
    if "user_answer" not in st.session_state:
        st.session_state.user_answer = None

    # Generate new question only when needed
    if "generate_new" not in st.session_state:
        st.session_state.generate_new = True

    if st.session_state.generate_new:
        st.session_state.answer_submitted = False
        st.session_state.user_answer = None

        quiz_movies = movies.dropna(subset=['vote_average', 'release_date', 'cast', 'crew']).sample(4).reset_index(drop=True)
        correct_movie = quiz_movies.iloc[0]
        qtypes = ["rating", "release_year", "cast", "director", "true_false"]
        qtype = random.choice(qtypes)

        q_data = {"qtype": qtype, "movies": quiz_movies, "correct_movie": correct_movie}

        if qtype == "rating":
            q_data["question"] = "❓ Which movie has the highest TMDB rating?"
            q_data["options"] = quiz_movies['title'].tolist()
            q_data["correct"] = quiz_movies.loc[quiz_movies['vote_average'].idxmax(), 'title']

        elif qtype == "release_year":
            real_year = correct_movie['release_date'][:4]
            years = [real_year] + [str(int(real_year) + i) for i in [-5, -2, 2]]
            random.shuffle(years)
            q_data["question"] = f"📅 When was **{correct_movie['title']}** released?"
            q_data["options"] = years
            q_data["correct"] = real_year

        elif qtype == "cast":
            cast_list = correct_movie['cast'].split(',')[:3]
            correct_actor = random.choice(cast_list)
            all_cast = set(','.join(movies['cast'].dropna()).split(','))
            wrong = random.sample([c for c in all_cast if c not in cast_list], 3)
            options = [correct_actor] + wrong
            random.shuffle(options)
            q_data["question"] = f"👥 Who was in **{correct_movie['title']}**?"
            q_data["options"] = options
            q_data["correct"] = correct_actor

        elif qtype == "director":
            director = correct_movie['crew']
            all_directors = list(set(movies['crew'].dropna().unique()))
            wrong = random.sample([d for d in all_directors if d != director], 3)
            options = [director] + wrong
            random.shuffle(options)
            q_data["question"] = f"🎬 Who directed **{correct_movie['title']}**?"
            q_data["options"] = options
            q_data["correct"] = director

        elif qtype == "true_false":
            rating = correct_movie['vote_average']
            threshold = random.choice([6.0, 7.5, 8.5])
            statement = f"🎬 **{correct_movie['title']}** has a rating above {threshold}."
            q_data["question"] = f"✅ True or False: {statement}"
            q_data["options"] = ["True", "False"]
            q_data["correct"] = "True" if rating > threshold else "False"

        st.session_state.quiz_data = q_data
        st.session_state.generate_new = False

    # -------------------------
    # Show the current question
    # -------------------------
    q = st.session_state.quiz_data
    st.subheader(q["question"])
    user_answer = st.radio("Choose one:", q["options"], index=None)

    # Save selection
    if user_answer is not None:
        st.session_state.user_answer = user_answer

    # Submit Button
    submit = st.button("✅ Submit Answer")
    if submit and st.session_state.user_answer is not None:
        st.session_state.answer_submitted = True

    # Show result after submission
    if st.session_state.answer_submitted:
        if st.session_state.user_answer == q["correct"]:
            st.success("🎉 Correct!")
        else:
            st.error(f"❌ Incorrect. Correct answer: **{q['correct']}**")

    # Next Question
    if st.button("🔄 Next Question"):
        st.session_state.generate_new = True
        st.rerun()













