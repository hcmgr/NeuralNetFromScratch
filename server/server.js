const http = require('http')
const express = require('express')
const app = express()
require('dotenv').config()

const server = http.createServer(app)
server.listen(process.env.PORT || 80, () => {
    console.log(`Server running on ${process.env.PORT || 80}`)
})

app.use(express.static(__dirname + "/public"))

