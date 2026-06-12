---
layout: post
title: "What I learned building a 300-article knowledge base for a naval IT environment"
date: 2026-05-05
categories: knowledge-management
---

I spent years on nuclear submarines before transitioning to civilian IT work. On a submarine, procedure compliance is the difference between a safe ship and a very bad day. We internalize it quickly and there are serious consequences to going outside approved procedures. Every evolution has a written checklist and series of steps that have to be taken. Any departure from that procedure must be approved from powers rivaling the power of a deity. The culture of "if it isn't written down, it didn't happen" gets into your bones.

When I took over the IT Support Center at a naval shipyard engaged in nuclear maintenance, and found a knowledge library with a few outdated articles and most of the institutional knowledge living in people's heads, it bothered me more than it might have bothered someone with a different background. I knew what happens when the person who carries all that implicit knowledge is suddenly unavailable.

Over the past couple of years I've built the library out to over 300 articles. Here's what that actually looked like.

---

## The knowledge extraction problem is harder than it sounds

The hardest part wasn't writing. It was figuring out what to write *about*.

In any technical environment, the most valuable knowledge is the stuff that nobody thinks to document because it feels obvious to the people who know it. The workaround that's been done the same way for four years. The error message that only appears on one legacy system. The process that involves three different organizations and requires you to CC a specific person or it goes nowhere.

Getting that out of people's heads and into a format someone else can use is a different skill than technical writing. You have to watch someone do a thing, or sit with them while they walk through a problem, and notice what they don't say — the steps they skip because they've internalized them so deeply they don't register as steps anymore.

I got better at asking "what would a new hire do if this happened on a Saturday and they're the only one here?" That question tends to surface the real procedure.

---

## Structure matters more than completeness

Early on I was trying to be comprehensive. Long articles, every edge case covered, lots of context. But the feedback was it was too complicated and need to be streamlined.

I switched to a tighter template: symptom, likely cause, steps, escalation path. Four sections. If it didn't fit in that structure, it either needed to be two articles or it wasn't a troubleshooting guide but a reference document.

A troubleshooting guide is meant to be something someone opens in the middle of a problem. It needs to get to the answer in under 30 seconds of reading. A reference document has a lot more context and could be something taught in a classroom before you have a problem. Still useful but not as good when you need instant answer on the phone with someone needing help. 

Once I had the template enforced, articles got shorter, more consistent, and more frequently used. The length of an article turned out to be inversely correlated with how often it got opened.

---

## Taxonomy is a political problem as much as a technical one

How you categorize things determines whether people can find them. That sounds obvious until you're three months in and your tagging system has become a gentle disaster because different people had different intuitions about what category a thing belonged to.

In an IT environment, you might have an issue that's simultaneously a hardware problem, a user account problem, and a software licensing problem depending on which angle you're looking at it from. People search for things using the vocabulary they already have, not the vocabulary your taxonomy assumed they'd have - this means we iterate after initial publishing with title and subtitles to ensure that common search terms return the expected and intended articles. 

The fix we landed on was tagging liberally and searching by symptom and error code along along with category. The category structure is mostly for browsing and reporting and tells leadership what kinds of issues we're handling. The tags and search terms make individual articles findable.

I also stopped trying to get the taxonomy perfect before adding articles. Perfect taxonomy is a way of avoiding the actual work. Good-enough taxonomy plus lots of articles beats perfect taxonomy plus fifty articles every time.

---

## The articles that don't exist are the real problem

At some point I started tracking the issues that came in repeatedly without a corresponding knowledge article. Usually, this was because they were messy or environment-specific or required institutional context that was hard to write down cleanly.

But the hard-to-document stuff is hard because it's the knowledge that only lives in experienced people's heads. Which means every time that person is unavailable, someone else is starting from scratch.

I now write or delegate the article immediately after the second time I see a problem come in that we didn't have good documentation for. This ensures we're not wasting effort on one-off issues. Once is a fluke but twice is a pattern. Additionally, for single-issue problems, we found it was sufficient to search our historical ticketing system for if the issue had occurred before and if it had been logged months or years before, this was the occasion to document it. 

---

## What I'd do differently

The biggest item I would start from the beginning would be better instrumentation - know which articles are getting opened, which are getting opened and then abandoned, and which have never been opened at all. Also learning about search terms and which terms get used the most when searching for items vs. which articles are used directly from linked sources. Usage data changes what you prioritize and looking at the library won't give you that but the logs will.

---

The 300-article number is real but it's also a little misleading as a metric. What matters is whether people can solve their problem without calling someone. We've gotten better at that. The library is part of why.

---

*Robert Piazza is a former submarine officer who runs an IT Support Center at a naval shipyard engaged in nuclear maintenance. He writes occasionally at robertpiazza.com and tinkers with home lab automation in Edmonds, WA.*
