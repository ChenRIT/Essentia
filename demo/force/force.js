var w = 1200,
    h = 400,
    fill = d3.scale.category10();

var vis = d3.select("#chart")
  .append("svg:svg")
  .attr("width", w)
  .attr("height", h);

var svg = d3.select("body")
            .append("svg:svg")
              .attr("width", w)
              .attr("height", h)
              .style("z-index", -10)
              .attr("id", "svg");

          svg.append('svg:defs').selectAll('marker')
              .data(['end'])
              .enter()
            .append('svg:marker')
              .attr({'id': "arrowhead",
                     'viewBox':'0 -5 10 10',
                     'refX': 22,
                     'refY': 0,
                     'orient':'auto',
                     'markerWidth': 8,
                     'markerHeight': 8,
                     'markerUnits': "strokeWidth",
                     'xoverflow':'visible'})
            .append('svg:path')
              .attr('d', 'M0,-5L10,0L0,5')
              .attr('fill', '#ccc');

d3.json("force.json", function(json) {
  var force = d3.layout.force()
      .charge(-450)
      .gravity(.2)
      .linkDistance(20)
      .nodes(json.nodes)
      .links(json.links)
      .size([w, h])
      .start();

  var link = vis.selectAll("line.link")
      .data(json.links)
    .enter().append("svg:line")
      .attr("class", "link")
      .attr("marker-end", "url(#arrowhead)")
      .style("stroke-width", function(d) { return Math.sqrt(d.value); })
      .attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  var node = vis.selectAll(".node")
      .data(json.nodes)
    .enter().append("g")
      .attr("class", "node")
      .call(force.drag);

  var circle = node.append("svg:circle")
      .attr("class", "node")
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; })
      .attr("r", function(d) {return d.size;})
      .style("fill", function(d) { return fill(d.group); });

  var title = node.append("svg:text")
      .attr("class", "node")
      .attr("text-anchor", "middle")
      .attr("x", function(d) { return d.x })
      .attr("y", function(d) { return d.y - 10})
      .style("font-size", "20")
      .text(function(d) { return d.name; });

  vis.style("opacity", 1e-6)
    .transition()
      .duration(1000)
      .style("opacity", 1);

  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
        .attr("y1", function(d) { return d.source.y; })
        .attr("x2", function(d) { return d.target.x; })
        .attr("y2", function(d) { return d.target.y; });

    node.attr("cx", function(d) { return d.x; })
        .attr("cy", function(d) { return d.y; });

    circle.attr("cx", function(d) { return d.x; })
          .attr("cy", function(d) { return d.y; });

    title.attr("x", function(d) { return d.x; })
         .attr("y", function(d) { return d.y - 10; });
  });
});
