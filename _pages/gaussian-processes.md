---
layout: page
title: Gaussian Processes
permalink: /blog/gaussian-processes/
description: My notes on Gaussian Processes, as I have been reading the classic book by Rasmussen and Williams [[pdf](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)]. This was a book I had wanted to read for a while, and I finally got around to reading it. The purpose of these notes is mostly to help me digest the material, but I hope these are useful to others as well.
---

{% assign parts = site.posts | where: "series", "gaussian-processes" | sort: "series_part" %}
{% if parts.size == 0 %}
  <p><em>This series is in progress &mdash; check back soon.</em></p>
{% else %}
  <ol class="series-list">
    {% for p in parts %}
      <li class="series-item">
        <h3 style="margin: 0.4rem 0 0.2rem;">
          Part {{ p.series_part }}: <a href="{{ p.url | relative_url }}">{{ p.title }}</a>
        </h3>
        <p class="post-meta" style="margin: 0 0 0.4rem;">{{ p.date | date: '%B %d, %Y' }} &middot; {{ p.content | number_of_words | divided_by: 180 | plus: 1 }} min read</p>
        <p style="margin: 0 0 1.2rem;">{{ p.description }}</p>
      </li>
    {% endfor %}
  </ol>
{% endif %}
