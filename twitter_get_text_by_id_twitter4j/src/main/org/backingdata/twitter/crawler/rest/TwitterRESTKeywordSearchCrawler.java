package main.org.backingdata.twitter.crawler.rest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import main.org.backingdata.twitter.crawler.util.CredentialObject;
import main.org.backingdata.twitter.crawler.util.PropertyUtil;
import main.org.backingdata.twitter.crawler.util.PropertyManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;
import twitter4j.json.DataObjectFactory;

/**
 * REST Cralwer of Twitter - by keyword(s)<br/>
 * It is possible to define:<br/>
 * - a list of keywords / hashtags to crawl<br/>
 * - a pool of Twitter API keys / tokens in order to speed up timeline cralwing<br/><br/>
 * 
 * As outcome of the crawling process, for each time-line to crawl a new .txt file is created containing one JSON Tweet 
 * per line (https://dev.twitter.com/overview/api/tweets). The name of such file has the following format:<br/>
 * 
 * *SHARED_STRING*_*KEYWORD*_upTo_*CURRENT_DATE*.txt<br/>
 * 
 * @author Francesco Ronzano
 *
 */
public class TwitterRESTKeywordSearchCrawler {

	private static Logger logger = LoggerFactory.getLogger(TwitterRESTKeywordSearchCrawler.class.getName());

	// Authentication
	private static List<String> consumerKey = new ArrayList<String>();
	private static List<String> consumerSecret = new ArrayList<String>();
	private static List<String> token = new ArrayList<String>();
	private static List<String> tokenSecret = new ArrayList<String>();

	// Full local path of a local text file containing a list of tweet terms (one per line)
	private static String fullPathOfTweetKeywordFile = "";

	// Terms
	private static Set<String> getKeywords = new HashSet<String>();

	// Output directory
	private static String outputDirPath = "";

	// Output format
	private static String outpuTweetFormat = "";

	// Language filter
	private static String languageFilter = "";

	private static String fileSharedName = "tweet_by_keyword";


	// Blocking queue for tweets to process
	private static Integer sleepTimeInMilliseconds = 5000;

	// Date formatter
	private static SimpleDateFormat sdf = new SimpleDateFormat("dd_M_yyyy__hh_mm_ss");

	public static void startCrawling() {

		sleepTimeInMilliseconds = new Integer( ((int) (5000d / new Double(consumerKey.size()))) + 250);

		ConfigurationBuilder cb = new ConfigurationBuilder();
		cb.setDebugEnabled(true).setJSONStoreEnabled(true);
		// Changed by me, using extended mode
		// See https://stackoverflow.com/questions/38717816/twitter-api-text-field-value-is-truncated
		cb.setTweetModeExtended(true);

		TwitterFactory tf = new TwitterFactory(cb.build());

		List<Twitter> twitterList = new ArrayList<Twitter>();

		for(int i = 0; i < consumerKey.size(); i++) {
			Twitter twitter = tf.getInstance();
			AccessToken accessToken = new AccessToken(token.get(i), tokenSecret.get(i));
			twitter.setOAuthConsumer(consumerKey.get(i), consumerSecret.get(i));
			twitter.setOAuthAccessToken(accessToken);
			twitterList.add(twitter);
		}

		try {

			Integer accountCredentialsId = 0;

			if(getKeywords != null && getKeywords.size() > 0) {
				for(String entry : getKeywords) {
					if(entry != null && !entry.equals("")) {

						Integer storedKeywordTweets = 0;

						File storageDir = new File(outputDirPath);
						PrintWriter twitterKeywordPW = null;
						String fileName = storageDir.getAbsolutePath() + "/" + fileSharedName + "_" + entry.replaceAll("\\W+", "") + "_upTo_" + sdf.format(new Date()) + ".txt";
						try {
							twitterKeywordPW = new PrintWriter(fileName, "UTF-8");
						} catch (FileNotFoundException e) {
							System.out.println("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
							e.printStackTrace();
						} catch (UnsupportedEncodingException e) {
							System.out.println("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
							e.printStackTrace();
						}

						System.out.println("\n-\nStart retrieving tweets with keyword: "  + entry);
						int retrievedTweetCounter = 0;
						
						Query query = new Query(entry);
						if(languageFilter != null && !languageFilter.trim().equals("")) {
							query.setLang(languageFilter);
						}


						int numberOfTweets = 300;
						long lastID = Long.MAX_VALUE;

						ArrayList<Status> statusList = new ArrayList<Status>();
						ArrayList<String> tweetsToStore = new ArrayList<String>();
						while(statusList.size () < numberOfTweets) {

							if(storedKeywordTweets >= numberOfTweets) {
								break;
							}

							if(numberOfTweets - statusList.size() > 100) {
								query.setCount(100);
							}
							else{ 
								query.setCount(numberOfTweets - statusList.size());
							}

							try {
								Twitter currentAccountToQuery =  twitterList.get(accountCredentialsId);
								logger.info("Queried account: "  + accountCredentialsId);
								accountCredentialsId = (accountCredentialsId + 1) % consumerKey.size();

								QueryResult result = currentAccountToQuery.search(query);

								Thread.currentThread().sleep(sleepTimeInMilliseconds);

								if(result.getTweets() == null || result.getTweets().size() == 0) {
									System.out.println("No tweets retrieved when paging - keyword: "  + entry);
									break;
								}
								else {
									System.out.println(result.getTweets().size() + " results found when paging (max ID: " + query.getMaxId() + ") - total results: " + (result.getTweets().size() + retrievedTweetCounter) + " - keyword: '"  + entry + "' - waiting " + sleepTimeInMilliseconds + " milliseconds...");
								}

								if(result.getTweets() != null && result.getTweets().size() > 0) {
									logger.info("Retrieved " + result.getTweets().size() + " tweets - keyword: "  + entry);
									for (int i = 0; i < result.getTweets().size(); i++) {
										Status status = result.getTweets().get(i);

										if(status != null && status.getCreatedAt() != null) {
											String msg = DataObjectFactory.getRawJSON(status);
											if(msg == null) {
												System.out.println("ERROR > INVALID TWEET RETRIEVED!");
												continue;
											}
											tweetsToStore.add(msg);
											retrievedTweetCounter++;
										}
									}
								}
								else {
									logger.info("Retrieved NO tweets - keyword: "  + entry);
								}

								logger.info("Gathered in total " + statusList.size() + " tweets - keyword: "  + entry);
								for (Status t: result.getTweets()) {
									if(t.getId() < lastID) {
										lastID = t.getId();
										query.setMaxId(lastID);
									}
								}
							}
							catch (TwitterException te) {
								te.printStackTrace();
								logger.info("ERROR: Couldn't connect: " + te.getMessage());
							}; 

							query.setMaxId(lastID-1);
							
							
							// Store to file
							if(tweetsToStore.size() > 199) {
								System.out.println("\nStoring " + tweetsToStore.size() + " tweets in " + outpuTweetFormat + " format:");
								int storageCount = 0;
								for(String tweet : tweetsToStore)  {
									
									if(tweet != null) {
										if(outpuTweetFormat.equals("tab")) {
											Status status = DataObjectFactory.createStatus(tweet);
											twitterKeywordPW.write(status.getId() + "\t" + ((status.getText() != null) ? status.getText().replace("\n", " ") : "") + "\n");
											storageCount++;
										}
										else {
											twitterKeywordPW.write(tweet + "\n");
											storageCount++;
										}
									}
									
								}
								
								tweetsToStore = new ArrayList<String>();
								System.out.println(storageCount + " tweet stored to file: " + fileName);
							}
							
							
							twitterKeywordPW.flush();
							
						}
						
						System.out.println("Tweets stored to file: " + fileName);
					}
				}
			}

		} catch (Exception e) {
			logger.info("Error generic: " + e.getMessage());
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		if(args == null || args.length == 0 || args[0] == null || args[0].trim().equals("")) {
			System.out.println("Please, specify the full local path to the crawler ptoperty file as first argument!");
			return;
		}

		File crawlerPropertyFile = new File(args[0].trim());
		if(crawlerPropertyFile == null || !crawlerPropertyFile.exists() || !crawlerPropertyFile.isFile()) {
			System.out.println("The path of the crawler ptoperty file (first argument) is wrongly specified > PATH: '" + ((args[0] != null) ? args[0].trim() : "NULL") + "'");
			return;
		}


		// Load information from property file
		PropertyManager propManager = new PropertyManager();
		propManager.setPropertyFilePath(args[0].trim());

		// Load credential objects
		System.out.println("Loading twitter API credentials from the property file at '" + args[0].trim() + "':");
		List<CredentialObject> credentialObjList = PropertyUtil.loadCredentialObjects(propManager);
		if(credentialObjList != null && credentialObjList.size() > 0) {
			for(CredentialObject credentialObj : credentialObjList) {
				if(credentialObj != null && credentialObj.isValid()) {
					consumerKey.add(credentialObj.getConsumerKey());
					consumerSecret.add(credentialObj.getConsumerSecret());
					token.add(credentialObj.getToken());
					tokenSecret.add(credentialObj.getTokenSecret());
				}
				else {
					System.out.println("      - ERROR > INVALID CREDENTIAL SET: " + ((credentialObj != null) ? credentialObj.toString() : "NULL OBJECT"));
				}
			}
		}

		// Load full path of keyword list file
		try {
			//There is a mistake in the settings/crawler.properties,
			// as PropertyManager.RESTtweetKeywordListPath, the language filter should be specified by tweetKeyword.outputFormat instead of tweetId.outputFormat
			String keywordlistFilePath = propManager.getProperty(PropertyManager.RESTtweetKeywordListPath);
			File tweetIDfile = new File(keywordlistFilePath);
			if(tweetIDfile == null || !tweetIDfile.exists() || !tweetIDfile.isFile()) {
				System.out.println("ERROR: keyword list input file path (property '" + PropertyManager.RESTtweetKeywordListPath + "')"
						+ " wrongly specified > PATH: '" + ((keywordlistFilePath != null) ? keywordlistFilePath : "NULL") + "'");
				if(tweetIDfile != null && !tweetIDfile.exists()) {
					System.out.println("      The file does not exist!"); 
				}
				if(tweetIDfile != null && tweetIDfile.exists() && !tweetIDfile.isFile()) {
					System.out.println("      The path does not point to a valid file!"); 
				}
				return;
			}
			else {
				fullPathOfTweetKeywordFile = keywordlistFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: keyword list input file path (property '" + PropertyManager.RESTtweetKeywordListPath + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Load full path of output directory
		try {
			String outputDirectoryFilePath = propManager.getProperty(PropertyManager.RESTtweetKeywordFullPathOfOutputDir);
			File outputDirFile = new File(outputDirectoryFilePath);
			if(outputDirFile == null || !outputDirFile.exists() || !outputDirFile.isDirectory()) {
				System.out.println("ERROR: output directory full path (property '" + PropertyManager.RESTtweetKeywordFullPathOfOutputDir + "')"
						+ " wrongly specified > PATH: '" + ((outputDirectoryFilePath != null) ? outputDirectoryFilePath : "NULL") + "'");
				if(outputDirFile != null && !outputDirFile.exists()) {
					System.out.println("      The directory does not exist!"); 
				}
				if(outputDirFile != null && outputDirFile.exists() && !outputDirFile.isDirectory()) {
					System.out.println("      The path does not point to a valid directory!"); 
				}
				return;
			}
			else {
				outputDirPath = outputDirectoryFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: output directory full path (property '" + PropertyManager.RESTtweetIDfullPathOfOutputDir + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Output format
		try {
			String outputFormat = propManager.getProperty(PropertyManager.RESTtweetKeywordOutputFormat);

			if(outputFormat != null && outputFormat.trim().toLowerCase().equals("json")) {
				outpuTweetFormat = "json";
			}
			else if(outputFormat != null && outputFormat.trim().toLowerCase().equals("tab")) {
				outpuTweetFormat = "tab";
			}
			else {
				outpuTweetFormat = "json";
				System.out.println("Impossible to read the '" + PropertyManager.RESTtweetKeywordOutputFormat + "' property - set to: " + outpuTweetFormat);
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.RESTtweetKeywordOutputFormat + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}


		// Language filter
		try {
			String langFilter = propManager.getProperty(PropertyManager.RESTtweetKeywordLimitByLanguage);

			if(langFilter != null) {
				languageFilter = langFilter;
			}
			else {
				languageFilter = "";
				System.out.println("Impossible to read the '" + PropertyManager.RESTtweetKeywordLimitByLanguage + "' property - Language filter not set");
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.RESTtweetKeywordLimitByLanguage + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		


		// Loading tweet keywords from file
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fullPathOfTweetKeywordFile)), "UTF-8"));

			String str;
			while ((str = in.readLine()) != null) {
				if(!str.trim().equals("")) {
					getKeywords.add(str.trim());
				}
			}

			in.close();
		}
		catch (Exception e) {
			System.out.println("Exception reading keywords from file: " +  e.getMessage() + " > PATH: '" + ((fullPathOfTweetKeywordFile != null) ? fullPathOfTweetKeywordFile : "NULL") + "'");
			return;
		}


		File storageDir = new File(outputDirPath);


		// Printing arguments:
		System.out.println("\n***************************************************************************************");
		System.out.println("******************** LOADED PARAMETERS ************************************************");
		System.out.println("   > Property file loaded from path: '" + ((args[0].trim() != null) ? args[0].trim() : "NULL") + "'");
		System.out.println("        PROPERTIES:");
		System.out.println("           - NUMBER OF TWITTER API CREDENTIALS: " + ((consumerKey != null) ? consumerKey.size() : "ERROR"));
		System.out.println("           - LANGUAGE FILTER: " + ((languageFilter != null) ? languageFilter : "ERROR"));
		System.out.println("           - PATH OF LIST OF KEYWORDS TO CRAWL: '" + ((fullPathOfTweetKeywordFile != null) ? fullPathOfTweetKeywordFile : "NULL") + "'");
		System.out.println("           - PATH OF CRAWLER OUTPUT FOLDER: '" + ((outputDirPath != null) ? outputDirPath : "NULL") + "'");
		System.out.println("           - OUTPUT FORMAT: '" + ((outpuTweetFormat != null) ? outpuTweetFormat : "NULL") + "'");
		System.out.println("   -");
		System.out.println("   NUMBER OF TWEET KEYWORDS / LINES READ FROM THE LIST: " + ((getKeywords != null) ? getKeywords.size() : "READING ERROR"));
		System.out.println("***************************************************************************************\n");		
		
		if(getKeywords == null || getKeywords.size() == 0) {
			System.out.println("Empty list of Tweet keyword to crawl > EXIT");
			return;
		}

		if(consumerKey == null || consumerKey.size() == 0) {
			System.out.println("Empty list of valid Twitter API credentials > EXIT");
			return;
		}
		
		System.out.println("<><><><><><><><><><><><><><><><><><><>");
		System.out.println("List of keywords to crawl:");
		int keywordCounter = 1;
		for(String keyword : getKeywords) {
			System.out.println(keywordCounter++ + " keyword: " + keyword);
		}
		System.out.println("<><><><><><><><><><><><><><><><><><><>");
		

		System.out.println("-----------------------------------------------------------------------------------");
		System.out.println("YOU'RE GOING TO USE " + ((consumerKey != null) ? consumerKey.size() : "ERROR") + " TWITTER DEVELOPER CREDENTIAL(S).");
		System.out.println("INCREASE YOUR CREDENTIAL NUMBER IN THE CONFIGURATION FILE IF YOU NEED TO INCREASE CRAWLING SPEED");
		System.out.println("-----------------------------------------------------------------------------------\n");
		
		
		try {
			Thread.sleep(4000);
		} catch (InterruptedException e) {
			/* Do nothing */
		}


		startCrawling();
	}

}
