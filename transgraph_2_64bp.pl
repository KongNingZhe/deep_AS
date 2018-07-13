$transgraph = '../Z_bysj/data/protein_transgraph';
$transgraph64bp = '../Z_bysj/protein_transgraph_64bp';
$hg19_path = '.././data/hg19.fa';
open gene,"<$hg19_path" or die$!;
open transgraph,"<$transgraph" or die$!;
open out,">$transgraph64bp" or die$!;
#open out2,">transgraph_N" or die$!;

$bp = 64;
$ban = 33;

print "###### hg19 hash making####\n";
$chr='chr1';
#$line = <transgraph>;	@words=split(/ +/,$line);$aa = join(';',@words); print "$aa";

while(<gene>)
{
    chomp;
      if ($_ =~ /^>(.*)/)
	  {
		$gene{$chr}=$seq;

        $chr = $1;
		if($1 eq 'chrM')
		{
			last;
		}
        $seq = "";
      }else
      {
                $seq .= $_;
      }
}
$gene{$chr} = $seq;
print "####### hg19 hash made####\n";
$max = 0;
while($line = <transgraph>)
{
    chomp $line;
	@words=split(/\s+/,$line);
    if($words[0] eq 'gene')
    {
        $mode = $words[1];
        $chr = $words[2];
        next;
    }
    else
    {
        $len = @words;
        $wordsbp[0] = 'N' x 64;
        #$wordsn[0] = '5' x 64;
        $wordsbp[$len - 4] = '0' x 64;
        #$wordsn[$len - 3] = '0' x 64;

        if($mode eq '+')
        {
            for($i = 1;$i<$len-4;$i++)
            {
                $wordsbp[$i]=substr($gene{'chr'.$chr},$words[$i+1]-$ban,$bp);
                #$number = substr($gene{$words[$len-1]},$words[$i]-$ban,$bp);
                #$number=~tr/atcgATCGN/123412345/;
                #$wordsn[$i]=$number;
            }
        }
        else
        {
            for($i = 1;$i<$len-4;$i+=2)
            {
                $seq = substr($gene{'chr'.$chr},$words[$i+1]-$ban,$bp);
                $seq =~tr/atcgATCG/tagcTAGC/;
                $seq = reverse $seq;
                $wordsbp[$i+1]=$seq;

                $seq = substr($gene{'chr'.$chr},$words[$i+2]-$ban,$bp);
                $seq =~tr/atcgATCG/tagcTAGC/;
                $seq = reverse $seq;
                $wordsbp[$i]=$seq;
                #$seq=~tr/atcgATCGN/123412345/;
                #$wordsn[$i]=$seq;
            }
        }
        print out "@wordsbp\t";
        @wordsbp = ();
        print out "$words[$len-1]\n";
    #    print out2 "@wordsn\t";
    #    print out2 "$words[$len-2]\n";
    }

}
