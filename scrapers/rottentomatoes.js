let request = require('superagent');
let LanguageDetect = require('languagedetect');
let lngDetector = new LanguageDetect();
require('superagent-charset')(request);
let cheerio = require('cheerio');
let async = require('async');
let R = require('ramda');

let baseUrl = 'https://www.rottentomatoes.com';
let topMoviesListUrl = 'https://www.rottentomatoes.com/top/bestofrt/';

// get html text from url, use superagent-charset for chinese issue
const getPage = (url) => new Promise((resolve) => {
    request.get(url)
    .end((err, sres) => {
        var html = sres.text;
        var $ = cheerio.load(html, {decodeEntities: false});
        resolve($);
    })
});

// get top movies urls
const getTopUrls = (topPage) => new Promise((resolve) => {
    let arr = []
    topPage('table.table').find('a.articleLink').each((i, el) => {
        arr.push(`${baseUrl}${el.attribs["href"]}/reviews/?type=user`);
    });
    resolve(arr);
});

// get reviews and sentiments from page
const scrapeSinglePage = (movieReviewUrl, pageNumber) => new Promise((resolve) => {
    let u = `${movieReviewUrl}&page=${pageNumber}`;
    // console.log(`you are scaping ${u}`);
    request.get(u)
    .end((err, sres) => {
        let html = sres.text;
        let $ = cheerio.load(html, {decodeEntities: false});
        // get labels first
        let labels = []
        $('div.review_table').find('span.fl')
        .each((i, el) => {
            stars = el.children.filter(n => n.data !== ' ');
            label = stars ? stars.length : 1
            labels.push(label)
        });
        // get review texts, push label to each review
        let reviews = []
        $('div.review_table').find('div.user_review')
        .each((i, el) => {
            if (labels[i] && el.children[2] && el.children[2].data) {
                reviews.push({
                    'Phrase': el.children[2].data,
                    'Sentiment': labels[i]
                })
            }
        });
        // we dont need too long reviews
        reviews = reviews.filter(rev => rev.Phrase.split(' ').length <= 50)
        // we only want review written in English
        reviews = reviews.filter(rev => lngDetector.detect(rev['Phrase'], 1)[0] && lngDetector.detect(rev['Phrase'], 1)[0][0] === 'english')
        console.log(`Got ${reviews.length} reviews on page ${pageNumber} for movie ${movieReviewUrl.replace(baseUrl, '').split('/')[2]}`)
        resolve(reviews);
    });
});

// scrape reviews for a movie, we will try go through first 100 review pages of that movie
const scrapeMovie = async (movie_url, idx, cb) => {
    console.log(`task ${idx} started!!`)
    reviews = [];
    for(let p=1; p<=100; p++) {
        reviews = reviews.concat(await scrapeSinglePage(movie_url, p));
        if (reviews.length >= 100) {
            reviews = reviews.splice(0, 100)
            break;
        }
    }
    // console.log(`we just got ${reviews.length} movies`)
    console.log(`task ${idx} finished!!`)
    cb(null, reviews)
}

// control the parallel parameter
const scrape = (urlList, limit) => new Promise((resolve, reject) => {
    let tasks = urlList.map((u, idx) => (cb)=> scrapeMovie(u, idx, cb));
    // console.log(`found ${urlList.length} tasks`)
    async.parallelLimit(tasks, limit, (err, result) => {
        if (err) {
            console.log(`fuck the err: ${JSON.stringify(err)}`)
        }
        resolve(result);
    })
});

(async () => {
    try {
        $tp = await getPage(topMoviesListUrl)
        moviesList = await getTopUrls($tp)
        // console.log(`just found ${moviesList.length} movies`)
        results = R.flatten(await scrape(moviesList, 10));
        require('fs').writeFile('../movie_review_sentiment_analysis/data/vset.json', JSON.stringify(results), () => {
            console.log('write file done')
        });
    } catch(err) {
        console.log(err)
    }
})();