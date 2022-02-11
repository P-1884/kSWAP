'Weds' files are just a spare copy of swap (from the 25th jab beta test) with which to do data exports while the original swap is still running. No major changes of code have/will be made to it, so can just use swap.db for the real thing.

'simul' files are taken from the beta test simultaneously, so the database and classification files should have the same classifications in (used to compare AWS swap with CSV-online swap).

'Exclude logged on': This is just to match the AWS swap for the beta test which had to exclude not-logged-on users. In future runs of spacewarps these classifications can be included.

'setalreadyseen3tofalse': Setting the value described in number [3] of the process_from_csv function to false.