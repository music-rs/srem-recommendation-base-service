
# SREM Recommendation base service

## 1. Extract music events' data:
Firstly, we want to retrieve events data of Facebook, Joinnus and Teleticket, so, open a terminal and run the following commands:

- Facebook Webscraping:  **(don't works, as Facebook adds new restrictions)**
 
	- Music events:

		  scrapy crawl eventsGeneralFB -a email="<email-user>" -a password="<password-user>" -a page="https://mbasic.facebook.com/events/" -a lang="es" -o EVENTS_FB.csv

	- Page's posts:
	
		  scrapy crawl fb -a email="barackobama@gmail.com" -a password="10wnyu31" -a page="DonaldTrump" -a date="2018-01-01" -a lang="it" -o Trump.csv

	- Page's comments:
	  
		  scrapy crawl comments -a email="<email-user>" -a password="<password-user>" -a page="DonaldTrump" -a date="2019-09-01" -a lang="es" -o DUMPFILE_PAGE.csv

	- Post's comments:
			
		  scrapy crawl comments -a email="<email-user>" -a password="<password-user>" -a post="https://mbasic.facebook.com/story.php?story_fbid=10163085717830725&id=153080620724&refid=17&_ft_=mf_story_key.10163085717830725%3Atop_level_post_id.10163085717830725%3Atl_objid.10163085717830725%3Acontent_owner_id_new.153080620724%3Athrowback_story_fbid.10163085717830725%3Apage_id.153080620724%3Astory_location.4%3Apage_insights.%7B%22153080620724%22%3A%7B%22page_id%22%3A153080620724%2C%22actor_id%22%3A153080620724%2C%22dm%22%3A%7B%22isShare%22%3A0%2C%22originalPostOwnerID%22%3A0%7D%2C%22psn%22%3A%22EntStatusCreationStory%22%2C%22post_context%22%3A%7B%22object_fbtype%22%3A266%2C%22publish_time%22%3A1567350720%2C%22story_name%22%3A%22EntStatusCreationStory%22%2C%22story_fbid%22%3A%5B10163085717830725%5D%7D%2C%22role%22%3A1%2C%22sl%22%3A4%2C%22targets%22%3A%5B%7B%22actor_id%22%3A153080620724%2C%22page_id%22%3A153080620724%2C%22post_id%22%3A10163085717830725%2C%22role%22%3A1%2C%22share_id%22%3A0%7D%5D%7D%7D%3Athid.153080620724%3A306061129499414%3A2%3A0%3A1569913199%3A-6686756765087402683&__tn__=%2AW-R#footer_action_list" -a date="2019-08-01" -a lang="es" -o DUMPFILE_POST.csv


- Teleticket events Webscraping: **(works!!! but only retrieve events' name)**
 
		scrapy crawl eventsTeleticket -o EVENTS_TELETICKET.csv
		
- Joinnus events Webscraping:  **(don't works)**
	
		scrapy crawl eventsJoinnus -o EVENTS_JOINNUS.csv

- Webscraping with scroll
	
		scrapy crawl spidyquotes -o ejemplo_scroll.csv

## 2. Run Flask server (Recommendation Service)
This service uses the music events' data stored in EVENTS_FB.csv and EVENTS_TELETICKET.csv of the previous step.

* Execute the following commands:

		export FLASK_APP=algoritmoRecomendacion.py
		flask run

* Find ":'(" and comment that line (that allows us to use music events' data outdated)