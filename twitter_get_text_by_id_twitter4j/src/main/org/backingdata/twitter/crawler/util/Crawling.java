package main.org.backingdata.twitter.crawler.util;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashSet;
import java.util.Set;

import javax.imageio.ImageIO;

import main.org.backingdata.twitter.crawler.util.model.StoreMediaOutput;

import twitter4j.MediaEntity;
import twitter4j.MediaEntity.Variant;
import twitter4j.Status;

public class Crawling {

	/**
	 * Given a tweet, store the images / videos, animated_gifs associated if any
	 * 
	 * @param tweet
	 * @param storageDir
	 * @param storeImage
	 * @param storeVideo
	 * @param storeGif
	 */
	public static StoreMediaOutput storeMedia(Status tweet, String storageDir, boolean storeImage, boolean storeVideo, boolean storeGif) {
		
		StoreMediaOutput smoReturn = new StoreMediaOutput("");
		
		String errorStr = "";
		
		if(tweet != null && tweet.getMediaEntities() != null && tweet.getMediaEntities().length > 0) {
			
			// Create storage dir
			String storageDirInt = tweet.getId() + "";
			if(storageDirInt != null && storageDirInt.length() > 4) storageDirInt = storageDirInt.substring(0, 4);
			
			File storagrDirIntFile = new File(storageDir + ((storageDir.endsWith(File.separator)) ? "" : File.separator) + storageDirInt);
			if(storagrDirIntFile == null || !storagrDirIntFile.exists() || !storagrDirIntFile.isDirectory()) {
				storagrDirIntFile.mkdir();
			}
			storageDirInt = storagrDirIntFile.getAbsolutePath();
			
						
			// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > '" + tweet.getText() + "'");

			MediaEntity[] medias = tweet.getMediaEntities();

			for(MediaEntity media : medias) {
				try {
					if(media != null && media.getType() != null) {

						String mediaURL = media.getMediaURL();
						if(mediaURL.lastIndexOf("/") != -1) mediaURL = mediaURL.substring(mediaURL.lastIndexOf("/") + 1);
						String mediaFileName = tweet.getId() + "_" + media.getId() + "_" + mediaURL;
						String mediaStoragePath = storageDirInt + ((storageDirInt.endsWith(File.separator)) ? "" : File.separator) + mediaFileName;

						if(storeImage && media.getType().toLowerCase().trim().equals("photo")) {

							// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > Storing photo: " + media.getMediaURL() + " as file: " + mediaStoragePath);

							// Photo media type
							String photoType = (mediaURL.lastIndexOf(".") != -1) ? mediaURL.substring(mediaURL.lastIndexOf(".") + 1) : "jpg";

							// Get and store image
							File PHOTO_fileCheck = new File(mediaStoragePath);

							if(PHOTO_fileCheck == null || !PHOTO_fileCheck.exists() || !PHOTO_fileCheck.isFile()) {
								try {
									URL imageURL = new URL(media.getMediaURL());
									BufferedImage saveImage = ImageIO.read(imageURL);
									ImageIO.write(saveImage, photoType, new File(mediaStoragePath));
									
									smoReturn.getImageFilePaths().add(mediaStoragePath);
								}
								catch(Exception e) {
									e.printStackTrace();
									errorStr += "TWEET ID " + tweet.getId() + " / MEDIA > Error storing photo: " + media.getMediaURL() + " > " + ((e.getMessage() != null) ? e.getMessage() : "-") + ".\n";
								}
							}
							else {
								// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > The photo: " + media.getMediaURL() + " already exists.");
							}
						}
						else if(storeVideo && media.getType().toLowerCase().trim().equals("video")) {
							
							Set<String> variantURLs = new HashSet<String>();
							int smallerBitRate = Integer.MAX_VALUE;
							String videoDownloadURL = null;
							for(Variant variant : media.getVideoVariants()) {
								if(variant != null && variant.getUrl() != null) {
									
									variantURLs.add(variant.getUrl());
									
									if(variant.getBitrate() > 0 && variant.getBitrate() <= smallerBitRate) {
										smallerBitRate = variant.getBitrate();
										mediaURL = variant.getUrl();
										if(mediaURL.lastIndexOf("/") != -1) mediaURL = mediaURL.substring(mediaURL.lastIndexOf("/") + 1);
										mediaFileName = tweet.getId() + "_" + media.getId() + "_" + mediaURL;
										mediaStoragePath = storageDirInt + ((storageDirInt.endsWith(File.separator)) ? "" : File.separator) + mediaFileName;
										videoDownloadURL= variant.getUrl();
									}
								}
							}
							
							if(videoDownloadURL != null) {
								// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > Storing video: " + videoDownloadURL + " (bit rate " + smallerBitRate + ") as file: " + mediaStoragePath);
								
								// Get and store video
								File VIDEO_fileCheck = new File(mediaStoragePath);

								if(VIDEO_fileCheck == null || !VIDEO_fileCheck.exists() || !VIDEO_fileCheck.isFile()) {
									try {
										URLConnection urlc = (new URL(videoDownloadURL)).openConnection();
										BufferedInputStream inputVideo = new BufferedInputStream(urlc.getInputStream());

										BufferedOutputStream outputVideo = new BufferedOutputStream(new FileOutputStream(mediaStoragePath));
										
										int read = 0;
										byte[] bytes = new byte[1024];
										while ((read = inputVideo.read(bytes)) != -1) {
											outputVideo.write(bytes, 0, read);
										}

										inputVideo.close();
										outputVideo.close();
										
										smoReturn.getVideoFilePaths().add(mediaStoragePath);
									}
									catch(Exception e) {
										e.printStackTrace();
										errorStr += "TWEET ID " + tweet.getId() + " / MEDIA > Error storing video: " + videoDownloadURL + " > " + ((e.getMessage() != null) ? e.getMessage() : "-") + ".\n";
									}
								}
								else {
									// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > The video: " + videoDownloadURL + " already exists.");
								}
							}
							else {
								// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > Impossible to download the video variant associated to the tweet ID: " + tweet.getId() + " > Variant URLs: " + variantURLs.toString());
								errorStr += "TWEET ID " + tweet.getId() + " / MEDIA > Impossible to download the video variant associated to the tweet ID: " + tweet.getId() + " > Variant URLs: " + variantURLs.toString() + ".\n";
							}
							
						}
						else if(storeGif && media.getType().toLowerCase().trim().equals("animated_gif")) {

							// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > Storing animated_gif: " + media.getMediaURL() + " as file: " + mediaStoragePath);
							
							Set<String> variantURLs = new HashSet<String>();
							int smallerBitRate = Integer.MAX_VALUE;
							String GIFDownloadURL = null;
							for(Variant variant : media.getVideoVariants()) {
								if(variant != null && variant.getUrl() != null) {
									
									variantURLs.add(variant.getUrl());
									
									if(variant.getBitrate() <= smallerBitRate) {
										smallerBitRate = variant.getBitrate();
										mediaURL = variant.getUrl();
										if(mediaURL.lastIndexOf("/") != -1) mediaURL = mediaURL.substring(mediaURL.lastIndexOf("/") + 1);
										mediaFileName = tweet.getId() + "_" + media.getId() + "_" + mediaURL;
										mediaStoragePath = storageDirInt + ((storageDirInt.endsWith(File.separator)) ? "" : File.separator) + mediaFileName;
										GIFDownloadURL= variant.getUrl();
									}
								}
							}
							
							if(GIFDownloadURL != null) {
								
								// Get and store animated_gif
								File GIF_fileCheck = new File(mediaStoragePath);

								if(GIF_fileCheck == null || !GIF_fileCheck.exists() || !GIF_fileCheck.isFile()) {
									try {
										URLConnection urlc = (new URL(GIFDownloadURL)).openConnection();
										BufferedInputStream inputGIF = new BufferedInputStream(urlc.getInputStream());

										BufferedOutputStream outputGIF = new BufferedOutputStream(new FileOutputStream(mediaStoragePath));
										
										int read = 0;
										byte[] bytes = new byte[1024];
										while ((read = inputGIF.read(bytes)) != -1) {
											outputGIF.write(bytes, 0, read);
										}

										inputGIF.close();
										outputGIF.close();
										
										smoReturn.getAnimated_gifFilePaths().add(mediaStoragePath);
									}
									catch(Exception e) {
										e.printStackTrace();
										errorStr += "TWEET ID " + tweet.getId() + " / MEDIA > Error storing animated_gif: " + GIFDownloadURL + " > " + ((e.getMessage() != null) ? e.getMessage() : "-") + ".\n";
									}
								}
								else {
									// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > The animated_gif: " + media.getMediaURL() + "  already exists.");
								}
							}
							else {
								// System.out.println("TWEET ID " + tweet.getId() + " / MEDIA > Impossible to download the animated_gif variant associated to the tweet ID: " + tweet.getId() + " > Variant URLs: " + variantURLs.toString());
								errorStr += "TWEET ID " + tweet.getId() + " / MEDIA > Impossible to download the animated_gif variant associated to the tweet ID: " + tweet.getId() + " > Variant URLs: " + variantURLs.toString() + ".\n";
							}
							
						}
					}
				}
				catch(Exception e) {
					e.printStackTrace();
				}
			}
		}
		
		
		smoReturn.setErrorStr(errorStr);
		
		return smoReturn;
	}

}
