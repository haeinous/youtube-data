# YouTube Data Collective (YTDC)

(way better readme coming soon I promise! I also need to debug some stuff and clean up the code)

The YouTube Data Collective (YTDC) crowdsources information about YouTube video demonetization, allowing creators to gain third-party visibility into how YouTube's policies are implemented. When users submit new data, YTDC supplements it with data from the YouTube API, conducts thumbnail image analysis with the Clarifai API, sentiment analysis on the text fields with Google Cloud's Natural Language API. YTDC has a custom search engine based on an inverted index where postings are stored in a linked list; results are ranked through tf–idf. D3's Force layout visualizes connections among videos, drawing from an undirected graph implemented with an adjacency list. Another chart allows users to compare video tags and uses a trie for autocomplete. YTDC launched with a seed data set of over 30,000 videos.

## Things I'm proud of:

### 1. Building a search engine from scratch


### 2. Autocomplete functionality using a trie

![alt text](https://lh3.googleusercontent.com/it9AXLJodcb8Kqbe5KqLal9z_QMyxhfLvdG6DMjz_sudQqZ4sNwWifyJ6zeSAwMEBqtKY7YRqVO3Iw=w1440-h780 "Autocomplete functionality")


### 3. Using a multiset to return random videos in O(1) time



### 4. The data model with lots of referential integrity

[link to data model](https://www.lucidchart.com/invitations/accept/ed10e5ff-b073-4515-bcf0-0cefe057c7f7)

## Things I learned:
* I like data structures
* There are myriad and obscure ways that APIs can throw curveballs
* It's amazing that Google is able to show me relevant results in way less than a second
* Linear algebra
* I'm so, so excited for my new life as a software engineer

## Tech stack
* Languages: Python, JavaScript, HTML/CSS
* Frameworks: SQLAlchemy, PostgreSQL, Flask/Jinja, jQuery, Bootstrap
* Libraries: D3, Chart.js, Natural Language Toolkit (NLTK), NumPy
* APIs: [YouTube Data API](https://developers.google.com/youtube/v3/docs/), [Clarifai](https://clarifai.com/developer/guide/) for image analysis, and Google Cloud's [Natural Language Processing](https://cloud.google.com/natural-language/docs/basics) for sentiment analysis.

## Next steps (if I were to take them)
I'm not deploying ytdc because it has way too many bugs, and I have too many other projects I want to work on. If I were to improve my app, I'd do the following:

* 100% test coverage
* Enhance search functionality by:

** Incorporate video/channel tags and descriptions into the inverted index (the resulting index was too large for to pickle in memory for my computer despite increasing the maximum recursion)

* Increase search efficiency by:
** Implementing a priority queue using a heap to reduce time complexity to O(log n)
** 

## Gratitude:

Note: The color scheme is actually [Teespring's rebrand colors](http://teespring.com/style-guide)

(for so many people, coming soon!)


