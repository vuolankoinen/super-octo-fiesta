import org.jsoup.Jsoup;
import org.jsoup.select.Elements;
import org.jsoup.nodes.Element;
import org.jsoup.nodes.Document;
import java.io.IOException;


class ahmatti {
    public static void main(String[] args) {
	System.out.println("Hei! Ladataan sivua, odota hetki. :)");
	try {
	    Document yle = Jsoup.connect("http://yle.fi/uutiset/").get();
	    Elements uutiset = yle.select("#initialState");
	    System.out.println("Ylen uutisia: \n" + uutiset.toString());
	} catch (IOException e) {
	    System.out.println("Virhe! " + e.getMessage());
	}
    }
}
