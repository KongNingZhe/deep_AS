use strict ;
#use warnings ;
open IN,"<../Z_bysj/data/protein_transcript" or die$!;
open OUT,">../Z_bysj/data/protein_transcript_clean" or die$!;

my $clean_s_len = 300;
my $clean_e_len =10;

while(my $line = <IN>)
{
	chomp $line;
	my @words=split('\t',$line);
	if($line eq 'gene')
	{
		print OUT "gene\n";
	}
	else
	{
		my @word = split(' ',$words[0]);
		my $word_len = @word;
		my $ifpass = 0;
		my $ifpos = 0;
		if(($word[2]-$word[1]) > 0)
		{
			$ifpos =1;
		}
		if($word_len <= $clean_s_len)
		{ 
			for(my $wordi = 2;$wordi < $word_len-1;$wordi += 2)
			{
				if(abs(($word[$wordi]-$word[$wordi-1])) < $clean_e_len)
				{
					$ifpass = 1;
				}
				if($ifpos)
				{
					if(($word[$wordi]-$word[$wordi-1]) < 0)
					{
						$ifpass = 1;
					}
				}
				else
				{
					if(($word[$wordi]-$word[$wordi-1]) > 0)
					{
						$ifpass = 1;
					}
				}
			}
		}
		else 
		{
			$ifpass = 1;
		}

		if ($ifpass)
		{
			next;
		}
		print OUT"$words[0]\t$words[1]\t$words[2]\n";
	}
}