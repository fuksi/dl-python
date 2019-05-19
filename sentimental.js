const axios = require('axios')
const moment = require('moment')
const fs = require('fs')
const wait = require('wait-for-stuff')
const lineByLine = require('n-readlines')

var bitcoinFiles = 'data/bitconnect/'
var rippleFiles = 'data/ripple/'
var files = fs.readdirSync(rippleFiles)

files.forEach(file => {
    const liner = new lineByLine(rippleFiles + '/' + file);
    let line;
    while(line = liner.next()) {
        var data = {
            document: {
                type: "PLAIN_TEXT",
                content: line.toString('ascii')
            }
        }
        fetchAndStore(data, file)
        wait.for.time(0.1)
    }
});


function fetchAndStore(data, fileName) {
    let apiKey = ''
    let url = `https://language.googleapis.com/v1/documents:analyzeSentiment?key=${apiKey}`
    let filePath = 'sentimental/ripple/'+ fileName
    axios.post(url, data).then(res => {
        if (res.status == 200 && !!res.data) {
            var magnitude = res.data.documentSentiment.magnitude
            var score = res.data.documentSentiment.score * magnitude
            var text = "ripple," + fileName + "," + score*magnitude + "\r\n"
            console.log(text)
            fs.appendFileSync(filePath, text)
        }
    })
}
