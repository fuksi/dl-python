const axios = require('axios')
const moment = require('moment')
const fs = require('fs')
const wait = require('wait-for-stuff')

let start = moment('2018-01-01T00:00:00')
const periodEnd = moment('2019-01-01T00:00:00')
while (start < periodEnd) {
    end = start.clone().add(1, 'days')
    console.log(start.toDate())
    fetchAndStore(start.unix(), end.unix())
    wait.for.time(5)
    start = end
}

function fetchAndStore(startEpoch, endEpoch) {
    let url = `https://api.pushshift.io/reddit/search/submission/?q=Ripple&after=${startEpoch}&before=${endEpoch}` +
              '&subreddit=Ripple&author=&aggs=&metadata=true&frequency=hour' +
              '&advanced=false&sort=desc&domain=&sort_type=num_comments&size=10000'

    const fileName = moment(startEpoch * 1000).format('YYYY-MM-DD')
    const filePath = `data/ripple/${fileName}`
    if (!fs.existsSync(filePath)) {
        axios.get(url).then(res => {
            if (res.status == 200 && !!res.data) {
                const rows = res.data.data
                const test = res;
                const content = rows.map(i => i.title).join('\r\n')
                fs.writeFileSync(filePath, content)
            }
        })
    }
}
