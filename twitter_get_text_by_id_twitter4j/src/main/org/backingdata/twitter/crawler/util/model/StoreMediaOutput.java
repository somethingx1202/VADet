package main.org.backingdata.twitter.crawler.util.model;

import java.util.HashSet;
import java.util.Set;

public class StoreMediaOutput {
	
	private String errorStr = "";
	private Set<String> videoFilePaths = new HashSet<String>();
	private Set<String> imageFilePaths = new HashSet<String>();
	private Set<String> animated_gifFilePaths = new HashSet<String>();
	
	// Constructor
	public StoreMediaOutput(String errorStr) {
		super();
		this.errorStr = errorStr;
	}

	
	// Settes and getters
	public String getErrorStr() {
		return errorStr;
	}

	public void setErrorStr(String errorStr) {
		this.errorStr = errorStr;
	}

	public Set<String> getVideoFilePaths() {
		return videoFilePaths;
	}

	public void setVideoFilePaths(Set<String> videoFilePaths) {
		this.videoFilePaths = videoFilePaths;
	}

	public Set<String> getImageFilePaths() {
		return imageFilePaths;
	}

	public void setImageFilePaths(Set<String> audioFilePaths) {
		this.imageFilePaths = audioFilePaths;
	}

	public Set<String> getAnimated_gifFilePaths() {
		return animated_gifFilePaths;
	}

	public void setAnimated_gifFilePaths(Set<String> anumated_gifFilePaths) {
		this.animated_gifFilePaths = anumated_gifFilePaths;
	}
	
	
}
