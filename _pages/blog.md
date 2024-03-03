---
layout: default
permalink: /blog/
title: blog # Blog Chronicles: Waiting for the perfect moment between coffee sips and chaos... 
nav: true
nav_order: 1
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1 # The number of links before the current page
    after: 3 # The number of links after the current page
---

<<<<<<< HEAD
=======
Blog Chronicles: Waiting for the perfect moment between coffee sips and chaos...

>>>>>>> ab180cd59ef55f09f97af9797fd83105d3c84b2e
<div class="post">

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}

  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
  {% endif %}

{% if site.display_tags or site.display_categories %}
