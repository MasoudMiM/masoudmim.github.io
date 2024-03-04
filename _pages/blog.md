---
layout: default
permalink: /blog/
title: blog
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

Blog Chronicles: Written thoughts during the moments between coffee sips and chaos... 

<div class="post">

  {% if blog_name_size > 0 or blog_description_size > 0 %}
  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
  {% endif %}

  {% if post.redirect == blank %}
    <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
  {% elsif post.redirect contains '://' %}
    <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
    <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
      <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  {% else %}
    <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
  {% endif %}
 
</div>