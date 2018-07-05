# YouTube Data Collective (YTDC)

(way better readme coming soon I promise! I also need to debug some stuff and clean up the code)

The YouTube Data Collective (YTDC) crowdsources information about YouTube video demonetization, allowing creators to gain third-party visibility into how YouTube's policies are implemented. When users submit new data, YTDC supplements it with data from the YouTube API, conducts thumbnail image analysis with the Clarifai API, sentiment analysis on the text fields with Google Cloud's Natural Language API. YTDC has a custom search engine based on an inverted index where postings are stored in a linked list; results are ranked through tf–idf. D3's Force layout visualizes connections among videos, drawing from an undirected graph implemented with an adjacency list. Another chart allows users to compare video tags and uses a trie for autocomplete. YTDC launched with a seed data set of over 30,000 videos.

## Things I'm proud of:

### 1. Building a search engine from scratch


### 2. Autocomplete functionality using a trie

I wanted users to be able to compare demonetization percentages by video tag. The example below shows that videos tagged "Hillary Clinton" are more likely to be demonetized than those tagged "Donald Trump." (As noted above, a biased sample selection process and limitations in scraping make it impossible to infer anything meaningful from this data. But it is fun to look at.) I modeled the user experience on [Google Trends](https://trends.google.com/trends/explore?date=today%205-y&geo=US&q=%2Fm%2F0cqt90,%2Fm%2F0d06m5).

![alt text](https://lh4.googleusercontent.com/_1H2MKoNUGkDN87ydTvIbbIJAvgs6tu7RCp7ypXYUUjxc3U2Yo3dkZiRXG4eJTtCLYl9n5Qxapu4qg=w1440-h780 "Autocomplete functionality")

In order to build this, I would need autocomplete. Henry, a Hackbright instructor, advised me to look into tries (pronounced like "try" with an s; I only say this because two MIT computer science grads pronounced it incorrectly—one thought it was "tree" and the other thought it was "tree-ay").

#### Here's how I implemented it:
1. Create classes for the `Trie` object with an `add_words` method  and a `TrieNode` object with an `add_child` method and a `frequency` attribute for how many videos have used a particular tag. A particular sequence of characters constitutes a word (or tag) if its `frequency` is not 0 (the higher the frequency, the more common it is).
2. When seeding the database, create two tries: one with every single tag (there ended up being some 150,000+ unique video tags) and another for tags with more than three occurrences (this pared it down to around 16,000 unique video tags after culling ones like "christmas pond decorations" and "lebron invented barbershop").
3. Pickle both tries and store in memory so they don't have to be regenerated every time someone visits `tags.html`.
4. When a user visits `tags.html` (on pageload): (a) client sends an AJAX request for the most up-to-date tag trie to server; (b) server unpickles the concise tag trie, a dictionary (where key = character, value = another dictionary containing information on frequency and children); (c) server sends client a jsonified version of the trie.
5. As a user starts typing, a keyup JavaScript event triggers searches for all tags that begin with the typed input (see `getAllWords` function below) and orders them by frequency, displaying the most relevant to the user. 😲 **The fact that all of this happens so quickly (16,000 tags!) is a testament to this data structure's efficiency.** 😲

---
**This recursive function gets all words associated with a particular prefix (input).**
```javascript
  function getAllWords(node, prefix='') {
    let allTags = []
    if (node[1].freq !== 0) {
      allTags.push([prefix + node[0], node[1].freq]);
    }
    if (Object.entries(node[1].children).length !== 0) {
      for (let item of Object.entries(node[1].children)) {
        for (let tagItem of getAllWords([item[0], item[1]], prefix + node[0])) {
          allTags.push(tagItem);
        }
      }
    }
    allTags.sort((a, b) => (b[1] - a[1]));
    return allTags
  }
```
---

### 3. Using a multiset to return random videos in O(1) time



### 4. The data model with lots of referential integrity

![alt text](https://lh5.googleusercontent.com/VYNjJFxSbd6nYEsn3SIuvJRAeTvoiVfGIbCv3kUdbDdERJBzZjLjIeeU8WOfwmii7cQUbHiJfcCfOQ=w1417-h780-rw "Data model for the YouTube Data Collective project")


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
  * Incorporating video/channel tags and descriptions into the inverted index (the resulting index was too large for to pickle in memory for my computer despite increasing the maximum recursion)

* Increase search efficiency by:
  * Implementing a priority queue with a heap to reduce time complexity to O(log n)

## Gratitude:

Note: The color scheme is actually [Teespring's rebrand colors](http://teespring.com/style-guide)

(for so many people, coming soon!)


