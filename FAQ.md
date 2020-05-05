# What is this repository?

This repository contains simple python scripts to fit and plot
covid-19 data that is published daily by Johns Hopkins CSSE
(https://github.com/CSSEGISandData/COVID-19)

The scripts transparently fetch the remote data from the Johns Hopkins
github repository so they are super easy to run, modify, and
experiment with.

# But we have professional/expert models.

Yes, but sometimes (ahem IHME) they swing wildly from update to
update, or are already significantly off by the time they are updated.
They have been shown to be terrible predictors or right on, but there
is no way to know and hard to trust some of their predictions.

https://towardsdatascience.com/transparency-reproducibility-and-validity-of-covid-19-projection-models-78592e029f28

# But fitting a straight line is overly simple, what does that tell us?

Yes, but look at how linear the data is in places.  Tell me you know
what function will fit it better.  Will a hot spot flair up and drive
up the numbers for the next few days?  Will an existing hot spot begin
burning out and drive the trend downward? Look at how wildly the
expert IHME model has swung around (above analysis link.)  I don't
think the short term trend is knowable.  If we can't predict (in the
short term) if the plot is going to turn up or down, then a straight
line fit will be as stable and useful as anything.

Secondly, a linear fit is actually the best fit through April and at
least into early May.  When the data trend legitimately diverges from
a linear fit, it will be easy to spot.  The experts predict an
exponential fall off, but we currently do not know when that will
begin or what slope/rate it will follow.

# Your plots are alarmist and scary!

My intention is not to be political.  At least through April, a linear
fit was much more realistic than other models being shopped around.
It wasn't too many weeks ago 5k deaths in the USA seemed completely
alarmist.  Now in early May we are talking about 100k by the end of
the summer or maybe a lot more.  What will the picture look like in
June?

# Why spend the time on this.

I think many of us are obsessed with the situation (early may as I
write this) and making and updating plots is an outlet.

# Comments?

I would love to hear comments and ideas that are constructive and
informational.

# Daily plots

See the daily plots:

https://github.com/clolsonus/covid-19
