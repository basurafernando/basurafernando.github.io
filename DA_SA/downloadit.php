<?php
$file = $_GET["fn"];
$filelog = fopen("sa_da_time_access.log","a");
fwrite($filelog,date("m/d/y : H:i:s", time()) . " | " .  $file .  "  | IP :"  .  $_SERVER['REMOTE_ADDR'] .  "\n");
fclose($filelog);


if (file_exists($file)) {
    header('Content-Description: File Transfer');
    header('Content-Type: application/octet-stream');
    header('Content-Disposition: attachment; filename='.basename($file));
    header('Content-Transfer-Encoding: binary');
    header('Expires: 0');
    header('Cache-Control: must-revalidate');
    header('Pragma: public');
    header('Content-Length: ' . filesize($file));
    ob_clean();
    flush();
    readfile($file);
    exit;
}
?>
