require 'time'

desc 'create a new draft post'
task :post do
    title = ENV['TITLE']
    if title.nil? || title.strip.empty?
        puts "ERROR: Please provide a TITLE. Example: rake post TITLE='My Post Title'"
        exit(1)
    end

    slug = "#{Date.today}-#{title.downcase.gsub(/[^\w]+/, '-')}"

    file = File.join(
        File.dirname(__FILE__),
        '_posts',
        slug + '.md'
    )

    timestamp = Time.now.strftime('%Y-%m-%d %H:%M:%S %z')

    File.open(file, "w") do |f|
        f << <<-EOS
---
layout: post
title: #{title}
date: #{timestamp}
categories: tech
latex: true
---
EOS
    end
    puts "New post created successfully: #{file}"
end
