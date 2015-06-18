'use strict';

// Create the grid the user will use to draw numbers.
var dim = 400, bN = 8, bD = dim / bN;
var svg = d3.select('body').append('svg').attr({width: dim, height: dim})
var data = d3.range(bN * bN).map(function (d) {
    return {i: d % bN, j: Math.floor(d / bN), id: d, active: 0}
});

var blocks = svg.append('g').attr('class', 'blocks')
    .selectAll('rect').data(data).enter().append('rect')
    .attr({width: bD, height: bD})
    .attr('x', function (d) {
        return bD * d.i
    })
    .attr('y', function (d) {
        return bD * d.j
    })
    .style('stroke', 'rgba(255, 255, 255, 0.2)')

function updateBlock(g) {
    g.style('fill', function (d) {
        return d.active ? 'rgba(255, 255, 255, 1)' : 'black'
    })
}

var conf = d3.select('.confidences').selectAll('div')
    .data(d3.range(10)).enter()
    .append('div').text(function (d) {
        return d + ': '
    })

var bar = conf.append('div').attr('class', 'bg')
    .style({
        width: '60px',
        height: '20px',
        display: 'inline-block'
    })
bar.append('div').attr('class', 'fg')
    .style({
        width: '20%',
        height: '100%',
        'background-color': 'rgba(0, 0, 0, 0.8)'
    })
svg.on('mousedown', function () {
    svg.on('mousemove', function () {
        var p = d3.mouse(this)
        var i = Math.floor(p[0] / bD)
        var j = Math.floor(p[1] / bD)
        blocks.filter(function (d) {
            return d.i === i && d.j === j
        })
            .each(function (d) {
                d.active = 1
            })
            .call(updateBlock)
        updateStats()
    })
}).on('mouseup', function () {
    svg.on('mousemove', null)
    updateStats()
})

function updateStats() {
    var x = data.map(function (d) {
        return d.active
    })
    var max = -1, maxId = -1
    var res = net(x)
    res.forEach(function (d, i) {
        if (d > max) max = d, maxId = i
    })
    console.log('maxId', maxId)
    console.log('confidences', max)
    conf.data(res).select('.fg')
        .style('width', function (d) {
            return d3.round(d * 100, 2) + '%'
        })
    d3.select('.prediction').text(maxId)
    d3.select('.confidence').text(max)
}

d3.select('button.clear').on('click', resetBlocks)

function resetBlocks() {
    blocks.each(function (d) {
        d.active = 0
    }).call(updateBlock)
}

// The MLP code.

// [-0.5, 0.5)
var rand = function () {
    return Math.random() - 0.5
}

// input layer -> [0, 63] or 8x8 image.
var iLen = 64

// Let's have a single hidden layer with 25 nodes.
var hLen = 25
// Matrix of hidden layer weights. The plus + 1 is for the input bias term.
var hW = d3.range(hLen).map(function () {
    return d3.range(iLen + 1).map(rand)
});

// We'll also have 10 output nodes, one for each digit [0, 9].
var oLen = 10
// Matrix of output layer weights. The plus + 1 is for the input bias term.
var oW = d3.range(oLen).map(function () {
    return d3.range(hLen + 1).map(rand)
});

// The sigmoid function. Our activation function.
function sig(x) {
    return 1 / (1 + Math.pow(Math.E, -x))
}
// The derivative of the sigmoid function. This function isn't actually used
// directly since we've already computed `sig` by the time we need `sig_prime`.
// We leave it here simply for reference.
function sig_prime(x) {
    return sig(x) * ( 1 - sig(x) )
}

function dot(a, b) {
    var res = 0, l = a.length
    for (var i = 0; i < l; i++) res = res + a[i] * b[i]
    return res
}

// Perform a forward pass of the network.
function forward(x) {
    // Calculate the hidden layer activations.
    var hA = [] // Activations for each hidden layer node.
    // For each hidden layer node...
    for (var i = 0; i < hW.length; i++) {
        // Calculate the activation.
        hA[i] = sig(dot(x, hW[i]) + hW[i][iLen] /* bias weight */)
    }
    var oA = []
    // Calculate the output layer activations.
    for (var i = 0; i < oW.length; i++) {
        // Calculate the activation.
        oA[i] = sig(dot(hA, oW[i]) + oW[i][hLen] /* bias weight */)
    }
    return {hA: hA, oA: oA}
}

// Update the weights by training the network.
function train(x, t, n) {
    // The node activations.
    var r = forward(x), hA = r.hA, oA = r.oA
    // Deltas for the output nodes.
    var oDel = []
    // For each output node...
    for (var i = 0; i < oLen; i++) {
        // Calculate the deltas from the errors.
        oDel[i] = (t[i] - oA[i]) * oA[i] * (1 - oA[i])
    }
    // Deltas for hidden layer.
    var hDel = []
    // For each hidden layer node...
    for (var i = 0; i < hLen; i++) {
        hDel[i] = 0
        // For each output layer node...
        // Collect all the deltas that fed out of this hidden layer node.
        for (var j = 0; j < oLen; j++) hDel[i] += oDel[j] * oW[j][i]
        hDel[i] = hDel[i] * hA[i] * ( 1 - hA[i] )
    }

    // Update the weights.
    // For each output node...
    for (var i = 0; i < oLen; i++) {
        for (var j = 0; j < hLen; j++) oW[i][j] += n * oDel[i] * hA[j]
        oW[i][hLen] += n * oDel[i] // Bias.
    }
    for (var i = 0; i < hLen; i++) {
        for (var j = 0; j < iLen; j++) hW[i][j] += n * hDel[i] * x[j]
        hW[i][iLen] += n * hDel[i] // Bias.
    }
}

function net(x) {
    return forward(x).oA
}

d3.text('training-data.csv', function (err, data) {
    if (err) throw err
    var ts, te, idx
    data = d3.csv.parseRows(data, function (row) {
        return row.map(Number)
    })
        .map(function (d) {
            // Expected output.
            var t = d3.range(10).map(function () {
                return 0
            })
            t[d[0]] = 1
            return {x: d.slice(1), t: t, label: d[0]}
        })
    ts = Date.now()
    for (var i = 0; i < 1e4; i++) {
        idx = i % data.length
        train(data[idx].x, data[idx].t, 0.1)
    }
    te = Date.now()
    console.log('time to train', te - ts)
})

