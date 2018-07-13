use strict ;
#use warnings ;
open IN,"<../Z_bysj/data/protein_transcript" or die$!;
#open IN,"<../Z_bysj/data/test" or die$!;
open OUT,">../Z_bysj/data/protein_transgraph" or die$!;

#pointhash 存每个节点的地址
my %pointshash;
#pathhash 放真实hash
my %pathhash;

my $max =1000;

my @path;
sub node {
	my ($value1,$value2) = @_ ;
	#my %hash = ();
	my $self = {
		'child' => undef,
		'value1' => $value1,
		'value2' => $value2,
		};
	return $self;
}

sub addnode
{
	my ($pre,$beh1,$beh2) = @_;
	my$beh = join("_",($beh1,$beh2));
	if (exists (${$pre->{child}}{$beh}))
	{
		return  ${$pre->{child}}{$beh};
	}
	else 
	{
		${$pre->{child}}{$beh} = $pointshash{$beh};
		return ${$pre->{child}} {$beh};
	}

}

sub addpath
{
	my ($point,@seq)=@_;
	my $num=@seq;
	for(my $i = 1;$i<$num;$i += 2)
	{
		$point = addnode($point,$seq[$i],$seq[$i+1]);
	}
}

sub counttree
{
	my ($point) = @_;
	if($point)
	{
		if($point->{value} eq '#')
		{
			return 1;
		}
		else
		{
			return 0;
		}
	}

}


sub printtree
{
	my ($point) = @_;
	if($point)
	{
		push (@path,$point->{value1});
		push (@path,$point->{value2});
		if($point->{value1} eq '#')
		{
			print OUT "@path\t";
			shift(@path);
			pop(@path);
			if( exists ($pathhash{join(" ",@path)}))
			{
				print OUT "1";
			}
			else
			{
				print OUT "0";
			}
			print OUT "\n";
			unshift(@path,'@');
			push(@path,'#');
		}
		else 
		{
			foreach my $i (keys %{$point->{child}})
			{
				printtree(${$point->{child}}{$i});
			}
		}
		pop @path;
		pop @path;
	}
}
# my @array1 = ('@',1,2,3,6,'#');
# my @array2 = ('@',1,3,4,5,'#');
# my 
# my $root = &node('@');
# &addpath($root,@array1);
# &addpath($root,@array2);
# &printtree($root);
my $i = 0;
my $root = undef;
my $chr;
my $mode = '+';
while(my $line = <IN>)
{
	chomp $line;
	my @words=split('\t',$line);
	if($words[0] eq 'gene')
	{
		#print here
		if ($i)
		{
			print OUT "gene\t$mode\t$chr\n";
		}
		my $pathnum = 1;
		my $childs;
		foreach my $n_node (values %pointshash)
		{
			if($n_node->{value1} ne '#')
			{
				$childs = keys %{$n_node->{child}};
				$pathnum *= $childs;
				#print "$pathnum\n";
			}
		}
		if ($pathnum <=300)
		{
			&printtree($root);
		}
		
		#clear some hash or data delete the tree
		%pointshash = ();
		%pathhash = ();
		$root = &node('@','@');
		$i++;
		if($words[1])
		{
			$mode = $words[1];
		}
		if ($i % 100 == 0)
		{
			print "已经处理了$i个基因\n";
		}

	}
	else
	{
		$chr = $words[2];
		$pathhash{$words[0]} = 1;
		my @word = split(' ',$words[0]);
		push(@word,'#');
		my $word_len = @word;
		for(my $w =1; $w < $word_len; $w +=2 )
		{
			if(exists $pointshash{join("_",($word[$w],$word[$w+1]))})
			{
				next;
			}
			else
			{
				$pointshash{join("_",($word[$w],$word[$w+1]))} = &node($word[$w],$word[$w+1]);
			}
		}
		&addpath($root,@word);
	}
}