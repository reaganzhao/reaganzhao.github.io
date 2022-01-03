---
layout: post
title:  Playing Fast and Loose with RAM
image:
  feature: pepper_crop.png
date:   2018-11-18 16:22
tags:   class_readings
---

Coming roughly ten years after "the cloud" became part of the general lexicon, and a year or two after the [FAWN paper](https://dl.acm.org/citation.cfm?id=1629577), Ousterhout et al's 2011 [RAMCloud](https://cacm.acm.org/magazines/2011/7/109885-the-case-for-ramcloud/fulltext) presented a vision of a future that would test the bounds of what the words "distributed" and "system" meant up to then.  Conceived as an entirely DRAM (dynamic random-access &mdash; i.e. semiconductor-based &mdash; memory) centric system, RAMCloud imagines RAM as permanent, distributed storage at scale. Such a distributed system would seem to significantly tax prior expectations (at least what we've seen so far from the literature) about the degree of fault tolerance and coordination needed to make a distributed system function.

RAM has made an appearance in the literature preceding RAMCloud; Ousterhout et al note both FAWN (2009), which was flash-based, even showing the query rate economy plot from the original FAWN paper, but arguing that the economy of scale will shift ever further to the advantage of DRAM. They also cite [a 2008 interview with Jim Gray](https://www.infoq.com/news/2008/06/ram-is-disk), who saw the writing on the wall for HDD and SSD-intensive memory solutions, arguing that "Ram is the new disk." From the 1980s to 2010, the authors note, disk access speed did increase (by 50X), but during that same time, disk *capacity* increased by 10,000X.

This outpacing is precisely the case for RAMCloud; disks are useful, but only as backup, everything else needs to be instantly available at our fingertips. Moreover, in spite of it's novelty, RAMCloud relies on the concept of "buffered logging," a method of asynchronous replication that ultimately leverages our old favorite, the [Log-Structured File System](https://web.stanford.edu/~ouster/cgi-bin/papers/lfs.pdf).

Ousterhout et al imagine RAMCloud as part of a soon-to-be-plausible MapReduce ecosystem, enabling applications developers to leverage distributed computation in concert with distributed storage, thereby avoiding the expense and headache of the SQL-to-NoSQL scaling problem. One of the bottlenecks noted by the authors is the latency of remote procedure calls, which would need to be below a certain threshold (5-10 microseconds) to make RAMCloud practical, though with the open sourcing of gRPC that followed a few years later, this would appear to be less of an obstacle.

In a the follow-on paper ["Fast Crash Recovery in RAMCloud",](https://web.stanford.edu/~ouster/cgi-bin/papers/ramcloud-recovery.pdf) Osterhout and his student Ongaro (who together would go on to write Raft) go into greater detail about the specifics of their implementation. In some sense, RAMCloud is the Log-Structured File System, but modernized, distributed, and built such that all data is always held in DRAM. "Buffered logging" works by replicating the log across backup replicas, which periodically (when their buffers fill up) flush log segments to disk in a single transfer.

In other distributed systems, availability is achieved with techniques like sharding and weakened consistency guarantees, or by attempts to optimize the efficiency of synchronous replication. In RAMCloud however, availability is achieved by orchestrating expedient recovery, such that it is no slower than the latency that might result from, say, running Paxos (moreover, RAMCloud apparently also guarantees linearizability, though the explanation does not appear in this paper).

In RAMCloud, a coordinator assigns objects to tablets (consecutive key ranges) for storage. Masters autonomously decide how to configure replicas; though this is implemented to be first random (to avoid pathological behavior) and then selective (to avoid replicating in ways that would be less tolerant to fault, like in the same rack). Masters write out "wills" (this was quite cute) that describe "how a master's assets should be divided in the event of it's demise". During recovery then, the coordinator reads the "will" of the crashed master, partitions the recovery into pieces, and delegates them to the recovery masters. Recovery masters work independently and in parallel to retrieve their assigned log segments and serve them using backup storage. This results, as is clear in the experimental results, in surprising fast recovery (albeit in a smaller cluster) and good performance.