# YouTube Data Collective (YTDC)

(way better readme coming soon I promise! I also need to debug some stuff and clean up the code)

The YouTube Data Collective (YTDC) crowdsources information about YouTube video demonetization, allowing creators to gain third-party visibility into how YouTube's policies are implemented. When users submit new data, YTDC supplements it with data from the YouTube API, conducts thumbnail image analysis with the Clarifai API, sentiment analysis on the text fields with Google Cloud's Natural Language API. YTDC has a custom search engine based on an inverted index where postings are stored in a linked list; results are ranked through tf–idf. D3's Force layout visualizes connections among videos, drawing from an undirected graph implemented with an adjacency list. Another chart allows users to compare video tags and uses a trie for autocomplete. YTDC launched with a seed data set of over 30,000 videos.

## Things I'm proud of:
* Autocomplete functionality using a trie
![alt text]( "Autocomplete functionality")
* The really bad search engine I built from scratch! 
* My convoluted data model with lots of referential integrity
* The color scheme (which is actually [Teespring's rebrand colors](http://teespring.com/style-guide))

## Things I learned:
* I like data structures
* There are very many ways APIs can throw curveballs (that don't surface until the 10,000th call)
* It's amazing that Google is able to show me relevant results in way less than a second
* Linear algebra
* I'm so, so excited for my new life as a software engineer

## Tech stack
* Languages: Python, JavaScript, HTML/CSS
* Frameworks: SQLAlchemy, PostgreSQL, Flask/Jinja, jQuery, Bootstrap
* Libraries: D3, Chart.js, Natural Language Toolkit (NLTK), NumPy
* APIs: [YouTube Data API](https://developers.google.com/youtube/v3/docs/), [Clarifai](https://clarifai.com/developer/guide/) for image analysis, and Google Cloud's [Natural Language Processing](https://cloud.google.com/natural-language/docs/basics) for sentiment analysis.

## Gratitude:
(for so many people, coming soon!)
