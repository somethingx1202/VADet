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
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import main.org.backingdata.twitter.crawler.util.CredentialObject;
import main.org.backingdata.twitter.crawler.util.PropertyUtil;
import main.org.backingdata.twitter.crawler.util.PropertyManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import twitter4j.Paging;
import twitter4j.ResponseList;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;
import twitter4j.json.DataObjectFactory;

/**
 * REST Cralwer of Twitter timelines<br/>
 * It is possible to define:<br/>
 * - a list of time-lines to crawl<br/>
 * - a pool of Twitter API keys / tokens in order to speed up timeline cralwing<br/><br/>
 * 
 * As outcome of the crawling process, for each time-line to crawl a new .txt file is created containing one JSON Tweet 
 * per line (https://dev.twitter.com/overview/api/tweets). The name of such file has the following format:<br/>
 * 
 * *SHARED_STRING*_*ACCOUNT_NAME*__*ACCOUNT_ID*_upTo_*CURRENT_DATE*.txt<br/>
 * 
 * @author Francesco Ronzano
 *
 */
public class TwitterRESTAccountTimelineCrawler {

	private static Logger logger = LoggerFactory.getLogger(TwitterRESTAccountTimelineCrawler.class.getName());

	// Authentication
	private static List<String> consumerKey = new ArrayList<String>();
	private static List<String> consumerSecret = new ArrayList<String>();
	private static List<String> token = new ArrayList<String>();
	private static List<String> tokenSecret = new ArrayList<String>();
	
	// Full local path of a local text file containing a list of tweet terms (one per line)
	private static String fullPathOfTweetTimelineFile = "";
	
	// Timelines
	private static Map<String, Long> getTimelines = new HashMap<String,Long>();

	// Output directory
	private static String outputDirPath = "";

	// Output format
	private static String outpuTweetFormat = "";

	private static String fileSharedName = "twitter_timeline";

	// Blocking queue for tweets to process
	private static Integer sleepTimeInMilliseconds = 5000;

	// Date formatter
	private static SimpleDateFormat sdf = new SimpleDateFormat("dd_M_yyyy__hh_mm_ss");

	public static void startCrawling() {

		sleepTimeInMilliseconds = new Integer( ((int) (5000d / new Double(consumerKey.size()))) + 250);

		ConfigurationBuilder cb = new ConfigurationBuilder();
		cb.setDebugEnabled(true).setJSONStoreEnabled(true);

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

			if(getTimelines != null && getTimelines.size() > 0) {
				for(Map.Entry<String, Long> entry : getTimelines.entrySet()) {
					if(entry.getKey() != null && !entry.getKey().equals("") && entry.getValue() != null) {
						String accountName = entry.getKey();
						Long userId = entry.getValue();

						Integer storedUSerTweets = 0;
						
						File storageDir = new File(outputDirPath);
						PrintWriter twitterTimelinePW = null;
						String fileName = storageDir.getAbsolutePath() + "/" + fileSharedName + "_" + accountName + "_" + userId + "_upTo_" + sdf.format(new Date()) + ".txt";
						try {
							twitterTimelinePW = new PrintWriter(fileName, "UTF-8");
						} catch (FileNotFoundException e) {
							System.out.println("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
							e.printStackTrace();
						} catch (UnsupportedEncodingException e) {
							System.out.println("CANNOT OPEN FILE: " + fileName + " - Exception: " + e.getMessage());
							e.printStackTrace();
						}

						System.out.println("\n-\nStart retrieving tweets of user ID: "  + userId);
						// Paging
						Integer pageNum = 1;
						Integer elementsPerPage = 40;
						Boolean nextPage = true;
						
						ArrayList<String> tweetsToStore = new ArrayList<String>();
						int errorsCouter = 0;
						while(nextPage) {
							Paging pagingInstance = new Paging();
							pagingInstance.setPage(pageNum);
							pagingInstance.setCount(elementsPerPage);
							try {
								System.out.println("Retrieving tweets of user ID: "  + userId + ", page: " + pageNum + ". Tweets per page: " + elementsPerPage + ", already stored: " + storedUSerTweets);

								Twitter currentAccountToQuery =  twitterList.get(accountCredentialsId);
								logger.info("Queried account: "  + accountCredentialsId);
								accountCredentialsId = (accountCredentialsId + 1) % consumerKey.size();
								ResponseList<Status> timeline = currentAccountToQuery.getUserTimeline(entry.getKey(), pagingInstance);
								pageNum++;

								Thread.sleep(sleepTimeInMilliseconds);

								if(timeline != null && timeline.size() > 0) {
									logger.info("Retrieved " + timeline.size() + " tweets (user ID: "  + userId + ", page: " + (pageNum - 1) + ". Tweets per page: " + elementsPerPage + ")");
									Iterator<Status> statusIter = timeline.iterator();
									while(statusIter.hasNext()) {
										Status status = statusIter.next();
										if(status != null && status.getCreatedAt() != null) {
											String msg = DataObjectFactory.getRawJSON(status);
											if(msg == null) {
												System.out.println("ERROR > INVALID TWEET RETRIEVED!");
												continue;
											}
											tweetsToStore.add(msg);
											storedUSerTweets++;
										}
									}
								}
								else {
									logger.info("Retrieved NO tweets (user ID: "  + userId + ", page: " + (pageNum - 1) + ". Tweets per page: " + elementsPerPage + ", already retrieved: " + storedUSerTweets + ")");
									nextPage = false;
								}
							} catch (TwitterException e) {
								logger.info("Error while querying Twitter: " + e.getMessage());
								e.printStackTrace();
								if(++errorsCouter > 2) {
									nextPage = false;
								}
							}
						}
						
						// Store to file
						System.out.println("\nStoring " + tweetsToStore.size() + " tweets in " + outpuTweetFormat + " format:");
						int storageCount = 0;
						for(String tweet : tweetsToStore)  {
							
							if(tweet != null) {
								if(outpuTweetFormat.equals("tab")) {
									Status status = DataObjectFactory.createStatus(tweet);
									twitterTimelinePW.write(status.getId() + "\t" + ((status.getText() != null) ? status.getText().replace("\n", " ") : "") + "\n");
									storageCount++;
								}
								else {
									twitterTimelinePW.write(tweet + "\n");
									storageCount++;
								}
							}
							
						}
						
						twitterTimelinePW.flush();
						
						System.out.println(storageCount + " tweet stored to file: " + fileName);
						System.out.println("Execution terminated.");
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
			System.out.println("Please, specify the full local path to the crawler property file as first argument!");
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

		// Load full path of tweetID file
		try {
			String timelineListFilePath = propManager.getProperty(PropertyManager.RESTtweetTimelineListPath);
			File tweetIDfile = new File(timelineListFilePath);
			if(tweetIDfile == null || !tweetIDfile.exists() || !tweetIDfile.isFile()) {
				System.out.println("ERROR: account IDs input file path (property '" + PropertyManager.RESTtweetTimelineListPath + "')"
						+ " wrongly specified > PATH: '" + ((timelineListFilePath != null) ? timelineListFilePath : "NULL") + "'");
				if(tweetIDfile != null && !tweetIDfile.exists()) {
					System.out.println("      The file does not exist!"); 
				}
				if(tweetIDfile != null && tweetIDfile.exists() && !tweetIDfile.isFile()) {
					System.out.println("      The path does not point to a valid file!"); 
				}
				return;
			}
			else {
				fullPathOfTweetTimelineFile = timelineListFilePath;
			}
		} catch (Exception e) {
			System.out.println("ERROR: account IDs input file path (property '" + PropertyManager.RESTtweetTimelineListPath + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Load full path of output directory
		try {
			String outputDirectoryFilePath = propManager.getProperty(PropertyManager.RESTtweetTimelineFullPathOfOutputDir);
			File outputDirFile = new File(outputDirectoryFilePath);
			if(outputDirFile == null || !outputDirFile.exists() || !outputDirFile.isDirectory()) {
				System.out.println("ERROR: output directory full path (property '" + PropertyManager.RESTtweetTimelineFullPathOfOutputDir + "')"
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
			System.out.println("ERROR: output directory full path (property '" + PropertyManager.RESTtweetTimelineFullPathOfOutputDir + "')"
					+ " wrongly specified - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Output format
		try {
			String outputFormat = propManager.getProperty(PropertyManager.RESTtweetTimelineOutputFormat);

			if(outputFormat != null && outputFormat.trim().toLowerCase().equals("json")) {
				outpuTweetFormat = "json";
			}
			else if(outputFormat != null && outputFormat.trim().toLowerCase().equals("tab")) {
				outpuTweetFormat = "tab";
			}
			else {
				outpuTweetFormat = "json";
				System.out.println("Impossible to read the '" + PropertyManager.RESTtweetTimelineOutputFormat + "' property - set to: " + outpuTweetFormat);
			}

		} catch (Exception e) {
			System.out.println("ERROR: output format (property '" + PropertyManager.RESTtweetTimelineOutputFormat + "') - exception: " + ((e.getMessage() != null) ? e.getMessage() : "NULL"));
			return;
		}

		// Loading tweet keywords from file
		// Per line ACCOUNT NAME <TAB> ACCOUNT_ID_LONG
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(new File(fullPathOfTweetTimelineFile)), "UTF-8"));

			String str;
			while ((str = in.readLine()) != null) {
				if(!str.trim().equals("")) {
					try {
						String[] strDiv = str.trim().split("\t");
						
						getTimelines.put(strDiv[0], Long.valueOf(strDiv[1].trim()));
					}
					catch(Exception e) {
						e.printStackTrace();
						System.out.println("Impossible to parse the account line:' " + str + "' from file: '" + fullPathOfTweetTimelineFile + "'");
					}
				}
			}

			in.close();
		}
		catch (Exception e) {
			System.out.println("Exception reading tweet accounts from file: " +  e.getMessage() + " > PATH: '" + ((fullPathOfTweetTimelineFile != null) ? fullPathOfTweetTimelineFile : "NULL") + "'");
			return;
		}


		File storageDir = new File(outputDirPath);


		// Printing arguments:
		System.out.println("\n***************************************************************************************");
		System.out.println("******************** LOADED PARAMETERS ************************************************");
		System.out.println("   > Property file loaded from path: '" + ((args[0].trim() != null) ? args[0].trim() : "NULL") + "'");
		System.out.println("        PROPERTIES:");
		System.out.println("           - NUMBER OF TWITTER API CREDENTIALS: " + ((consumerKey != null) ? consumerKey.size() : "ERROR"));
		System.out.println("           - PATH OF LIST OF ACCOUNT ID TO CRAWL: '" + ((fullPathOfTweetTimelineFile != null) ? fullPathOfTweetTimelineFile : "NULL") + "'");
		System.out.println("           - PATH OF CRAWLER OUTPUT FOLDER: '" + ((outputDirPath != null) ? outputDirPath : "NULL") + "'");
		System.out.println("           - OUTPUT FORMAT: '" + ((outpuTweetFormat != null) ? outpuTweetFormat : "NULL") + "'");
		System.out.println("   -");
		System.out.println("   NUMBER OF TWEET ACCOUNT IDS / LINES READ FROM THE LIST: " + ((getTimelines != null) ? getTimelines.size() : "READING ERROR"));
		System.out.println("***************************************************************************************\n");		

		if(getTimelines == null || getTimelines.size() == 0) {
			System.out.println("Empty list of Tweet timelines to crawl > EXIT");
			return;
		}

		if(consumerKey == null || consumerKey.size() == 0) {
			System.out.println("Empty list of valid Twitter API credentials > EXIT");
			return;
		}
		
		System.out.println("<><><><><><><><><><><><><><><><><><><>");
		System.out.println("List of account to crawl timeline:");
		int timelineCountetr = 1;
		for(Entry<String, Long> timeplineInfio : getTimelines.entrySet()) {
			System.out.println(timelineCountetr++ + " NAME: " + timeplineInfio.getKey() + " > ID: " + timeplineInfio.getValue());
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
