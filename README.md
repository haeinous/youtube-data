***(readme still being improved)***

# YouTube Data Collective (YTDC)

The YouTube Data Collective (YTDC) crowdsources information about YouTube video demonetization, allowing creators to gain third-party visibility into how YouTube's policies are implemented. When users submit new data, YTDC supplements it with data from the YouTube API, conducts thumbnail image analysis with the Clarifai API, sentiment analysis on the text fields with Google Cloud's Natural Language API. YTDC has a custom search engine based on an inverted index where postings are stored in a linked list; results are ranked through tf–idf. D3's Force layout visualizes connections among videos, drawing from an undirected graph implemented with an adjacency list. Another chart allows users to compare video tags and uses a trie for autocomplete. YTDC launched with a seed data set of over 30,000 videos.

## Things I'm proud of:

### 1. Building a search engine from scratch

I wanted users to be able to look up information for individual channels and videos. I considered Elasticsearch, Whoosh, and [PostgreSQL's built-in text search](https://www.postgresql.org/docs/9.5/static/textsearch.html) but ultimately decided to start from scratch (all the better to learn). I hunkered down with the highly readable ["Intro to Information Retrieval"](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf). And thanks to [Rebecca Weiss](http://stanford.edu/~rjweiss/public_html/IRiSS2013/text2/notebooks/tfidf.html) and [Christian S. Perone](http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) for creating easy-to-understand and sufficiently advanced online material to teach me enough linear algebra to fully understand the 
vector operations behind tf-idf.

### 2. Autocomplete functionality using a trie

I wanted users to be able to compare demonetization percentages by video tag. The example below shows that videos tagged "Hillary Clinton" are more likely to be demonetized than those tagged "Donald Trump." (As noted above, a biased sample selection process and limitations in scraping make it impossible to infer anything meaningful from this data. But it is fun to look at.) I modeled the user experience on [Google Trends](https://trends.google.com/trends/explore?date=today%205-y&geo=US&q=%2Fm%2F0cqt90,%2Fm%2F0d06m5).

Henry, a Hackbright instructor, encouraged me to implement autocomplete by looking into tries (pronounced like "try" with an s; I only say this because two MIT computer science grads pronounced it incorrectly—one thought it was "tree" and the other thought it was "tree-ay").

![alt text](https://raw.githubusercontent.com/haeinous/youtube-data/master/static/autocomplete.gif "Autocomplete functionality")

#### Here's how I implemented it:
1. Create classes for the `Trie` object with an `add_words` method  and a `TrieNode` object with an `add_child` method and a `frequency` attribute for how many videos have used a particular tag. A particular sequence of characters constitutes a word (or tag) if its `frequency` is not 0 (the higher the frequency, the more common it is).
2. When seeding the database, create two tries: one with every single tag (there ended up being some 150,000+ unique video tags) and another for tags with more than three occurrences (this pared it down to 16,000 unique video tags after culling ones like "christmas pond decorations" and "lebron invented barbershop").
3. Pickle both tries and store in memory so they don't have to be regenerated every time.
4. When a user visits `tags.html` (on pageload): (a) client sends an AJAX request for the most up-to-date tag trie to server; (b) server unpickles the concise tag trie, a dictionary (where key = character, value = another dictionary containing information on frequency and children); (c) server sends client a jsonified version of the trie.
5. As a user starts typing, a keyup JavaScript event triggers searches for all tags that begin with the typed input (see `getAllWords` function below) and orders them by frequency, displaying the most relevant to the user. 😲 **The fact that all of this happens so quickly (16,000 tags!) literally amazed me.** 😲

### 3. Using a multiset to return random videos in O(1) time

![alt text](https://github.com/haeinous/youtube-data/blob/master/static/multiset.gif "Sixteen random videos are returned in O(1) time out of more than 30,000 videos.")


### 4. A complex data model

![alt text](https://raw.githubusercontent.com/haeinous/youtube-data/master/static/ytdc%20-%20data_model.png "Data model for the YouTube Data Collective project")


## Things I learned:
* I like data structures
* There are myriad and obscure ways that APIs can throw curveballs
* It's amazing that Google is able to show me relevant results in way less than a second
* Linear algebra
* **I'm so, so excited for my new life as a software engineer**


## Next steps (if I were to take them)
I'm not deploying ytdc because it has way too many bugs, and I have too many other projects I want to work on. If I were to improve my app, I'd do the following:

* 100% test coverage
* Enhance search functionality by:
  * Incorporating video/channel tags and descriptions into the inverted index (the resulting index was too large for to pickle in memory for my computer despite increasing the maximum recursion)
  * Use a *positional* index in order to enable phrase queries, though this would increase space complexity
  * Use a B-tree and reverse B-tree to enable wildcard queries (e.g., words that start with "q" or end in "x")
* Increase search efficiency by:
  * Implementing a priority queue with a heap to reduce time complexity to `O(log n)`
* Use AJAX to implement infinite scroll

## Tech stack
* Languages: Python, JavaScript, HTML/CSS
* Frameworks: SQLAlchemy, PostgreSQL, Flask/Jinja, jQuery, Bootstrap
* Libraries: D3, Chart.js, Natural Language Toolkit (NLTK), NumPy
* APIs: [YouTube Data API](https://developers.google.com/youtube/v3/docs/), [Clarifai](https://clarifai.com/developer/guide/) for image analysis, and Google Cloud's [Natural Language Processing](https://cloud.google.com/natural-language/docs/basics) for sentiment analysis.

## Gratitude:

### People I know in person

I wouldn't have been able to do this without the support of my family—especially my mom and my [youngest brother Hyuckin/David](https://github.com/hdlim15) (who helped me build [@cooobot](https://github.com/haeinous/cooobot), my gateway to coding), as well as Youngin and Jungin/Sarah, whose love and support gave me the confidence to dive into this very new adventure. Much love and thanks to my best friend and roommate Hannah, who's known and supported me since freshman year of college, as well as her sister Rachel, whom I used to describe as the "only woman engineer I know."

Note: The color scheme is actually [Teespring's rebrand colors](http://teespring.com/style-guide)


### People I don't know in person

Also, I stumbled upon a computational linguistics class taught by one Chris Callison-Burch at UPenn. 
Working through the vector space model assignment (week 3) reminded me that I deeply enjoy (and am 
good at!) math. I'm excited to continue to work through the syllabus.

Another shoutout to Christopher Manning / Prabhakar Raghavan / Hinrich Schütze / Stanford (and maybe
Cambridge University Press?) for allowing the highly readable "Intro to Information Retrieval" textbook
to exist online for free.

(for so many people, coming soon!)


